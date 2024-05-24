import argparse
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_inference import TSPDataset as TSPDataset_Val
from dataset_inference import collate_fn as collate_fn_val
from model import subsequent_mask
from TSPmodel import TSPModel

class TSPModel_multistart(TSPModel):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def test_dataloader(self):
        self.test_dataset = TSPDataset_Val(self.cfg.val_data_path)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory=True
        )
        return test_dataloader

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        
        batch_size = tsp_tours.shape[0]
        node_size = tsp_tours.shape[1]
        src_original = src.clone()

        G = self.cfg.G

        src = src.unsqueeze(1).repeat(1, G, 1, 1).reshape(batch_size * G, node_size, 2) # [B * G, N, 2]
        tgt = torch.arange(G).to(src.device).unsqueeze(0).repeat(batch_size, 1).reshape(batch_size * G, 1) # [B * G, 1]
        
        visited_mask = torch.zeros(batch_size, G, 1, node_size, dtype = torch.bool, device = src.device) # [B, G, 1, N]
        visited_mask[:, torch.arange(G), :, torch.arange(G)] = True
        visited_mask = visited_mask.reshape(batch_size * G, 1, node_size)
        
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(self.cfg.node_size - 1):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                out = self.model.decode(memory, src, ys, tgt_mask)
                
                if self.cfg.comparison_matrix == "memory":
                    comparison_matrix = self.model.memory
                elif self.cfg.comparison_matrix == "encoder_lut":
                    comparison_matrix = self.model.encoder_lut
                elif self.cfg.comparison_matrix == "decoder_lut":
                    comparison_matrix = self.model.decoder_lut
                else:
                    assert False
                
                prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask, comparison_matrix)
                
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.squeeze(-1)
                
                visited_mask[torch.arange(batch_size * G), 0, next_word] = True
                
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1) # [B * G, N]
        
        predicted_tour_distances = self.get_tour_distance(src, ys) # [B * G, 1]
        predicted_tour_distances = predicted_tour_distances.reshape(batch_size, G)
        indices = torch.argmin(predicted_tour_distances, dim = -1) # [B]
        
        ys = ys.reshape(batch_size, G, node_size)
        ys = ys[torch.arange(batch_size), indices, :]
        
        from collections import Counter
        correct = []
        for idx in range(batch_size):
            rolled_ys = ys[idx].roll(shifts = -1)
            rolled_tour = tsp_tours[idx].roll(shifts = -1)
            edges_pred = set(tuple(sorted((ys[idx][i].item(), rolled_ys[i].item()))) for i in range(len(rolled_ys)))
            edges_tour = set(tuple(sorted((tsp_tours[idx][i].item(), rolled_tour[i].item()))) for i in range(len(rolled_tour)))
            counter_pred = Counter(edges_pred)
            counter_tour = Counter(edges_tour)

            common_edges = counter_pred & counter_tour
            correct.append(sum(common_edges.values()))
        
        correct = torch.tensor(correct)
        optimal_tour_distance = self.get_tour_distance(src_original, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src_original, ys)
        
        result = {
            "correct": correct.tolist(),
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.test_corrects.extend(result["correct"])
        self.test_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.test_predicted_tour_distances.extend(result["predicted_tour_distance"])
        
        return result
    
    def on_test_epoch_end(self):
        local_corrects = sum(self.test_corrects)
        local_optimal = sum(self.test_optimal_tour_distances)
        local_predicted = sum(self.test_predicted_tour_distances)
        
        self.test_corrects.clear()
        self.test_optimal_tour_distances.clear()
        self.test_predicted_tour_distances.clear()
        
        corrects = self.all_gather(local_corrects)
        optimal_tour_distances = self.all_gather(local_optimal)
        predicted_tour_distances = self.all_gather(local_predicted)
        
        if self.trainer.is_global_zero:
            correct = corrects.sum().item()
            total = self.cfg.node_size * len(self.test_dataset)
            hit_ratio = (correct / total) * 100
            
            opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
            mean_opt_gap = opt_gaps.mean().item() * 100
            self.hit_ratio = hit_ratio
            self.print(
                f"\ncorrect={correct}",
                f"\ntotal={total}",
                f"\nnode prediction(hit ratio) = {hit_ratio} %",
                f"\nmean_optimal_tour_distance = {optimal_tour_distances.sum() / len(self.test_dataset)}",
                f"\nmean_predicted_tour_distance = {predicted_tour_distances.sum() / len(self.test_dataset)}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
            )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.seed)
    
    tsp_model = TSPModel_multistart.load_from_checkpoint(cfg.resume_checkpoint)

    tsp_model.set_cfg(cfg)
    
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )
    
    trainer.test(tsp_model)