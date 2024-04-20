import time
from functools import reduce
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

from dataset_inference import TSPDataset, collate_fn, make_tgt_mask
from model import make_model, subsequent_mask, DecoderPositionalEncoding, DecoderCPE
from loss import SimpleLossCompute, LabelSmoothing

class TSPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = make_model(
            src_sz=cfg.node_size, 
            tgt_sz=cfg.decoder_output_size, 
            N=cfg.num_layers, 
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout,
            mode=cfg.mode,
            share_lut=cfg.share_lut,
            use_decoderCPE = cfg.use_decoderCPE
        )
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        
        self.set_cfg(cfg)
        
        self.test_corrects = []
        self.test_optimal_tour_distances = []
        self.test_predicted_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def test_dataloader(self):
        self.test_dataset = TSPDataset(self.cfg.val_data_path)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return test_dataloader
    
    def on_test_epoch_start(self):
        self.test_start_time = time.time()
        return

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        reversed_tsp_tours = batch["reversed_tsp_tours"]
        
        batch_size = tsp_tours.shape[0]
        node_size = tsp_tours.shape[1]
        src_original = src.clone()

        G = 20

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
                prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask, self.model.comparison_matrix)
                
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
        
        
        opt_gaps = (predicted_tour_distance - optimal_tour_distance) / optimal_tour_distance * 100
        with open('tsp20_inference1.txt', 'a') as f:
            for idx, cor, optgap in zip(indices.tolist(), correct.tolist(), opt_gaps.tolist()):
                f.write(f"{idx} {cor/self.cfg.node_size} {optgap}\n")
        
        """
        if self.trainer.is_global_zero:
            for idx in range(batch_size):
                print()
                print("predicted tour: ", ys[idx].tolist())
                print("optimal tour: ", tsp_tours[idx].tolist())
                print("opt, pred tour distance: ", optimal_tour_distance[idx].item(), predicted_tour_distance[idx].item())
                print("optimality gap: ", ((predicted_tour_distance[idx].item() - optimal_tour_distance[idx].item()) / optimal_tour_distance[idx].item()) * 100, "%")
                print("node prediction [hit ratio]: ", (correct[idx].item() / self.cfg.node_size) * 100 , "%")
                print()
        """
        
        return result
    
    def get_tour_distance(self, graph, tour):
        # graph.shape = [B, N, 2]
        # tour.shape = [B, N]

        shp = graph.shape
        gathering_index = tour.unsqueeze(-1).expand(*shp)
        ordered_seq = graph.gather(dim = 1, index = gathering_index)
        rolled_seq = ordered_seq.roll(dims = 1, shifts = -1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances
    
    def on_test_epoch_end(self):
        corrects = self.all_gather(sum(self.test_corrects))
        optimal_tour_distances = self.all_gather(sum(self.test_optimal_tour_distances))
        predicted_tour_distances = self.all_gather(sum(self.test_predicted_tour_distances))
        
        self.test_corrects.clear()
        self.test_optimal_tour_distances.clear()
        self.test_predicted_tour_distances.clear()
        
        if self.trainer.is_global_zero:
            correct = corrects.sum().item()
            total = self.cfg.node_size * len(self.test_dataset)
            hit_ratio = (correct / total) * 100
            
            opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
            mean_opt_gap = opt_gaps.mean().item() * 100
            validation_time = time.time() - self.test_start_time

            self.print(
                f"\ncorrect={correct}",
                f"\ntotal={total}",
                f"\nnode prediction(hit ratio) = {hit_ratio} %",
                f"\nmean_opt_gap = {mean_opt_gap}  %\n",
                "validation time={:.03f}".format(validation_time),

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
    
    # tsp_model = TSPModel(cfg)
    tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint)

    if cfg.use_decoderCPE:
        tsp_model.model.decoder_pe = DecoderCPE(cfg.d_model, cfg.dropout, cfg.node_size)
    else:
        tsp_model.model.decoder_pe = DecoderPositionalEncoding(cfg.d_model, cfg.dropout, cfg.node_size)

    tsp_model.set_cfg(cfg)
    
    # build trainer
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
