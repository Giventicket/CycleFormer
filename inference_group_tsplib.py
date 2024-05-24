import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

from dataset import TSPDataset, collate_fn, make_tgt_mask
from dataset_tsplib import TSPDataset as TSPDataset_Val
from dataset_tsplib import collate_fn as collate_fn_val

from model import make_model, subsequent_mask, PositionalEncoding_Circular
from loss import SimpleLossCompute, LabelSmoothing

import tsplib95
import csv

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class TSPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = make_model(
            src_sz=cfg.node_size, 
            enc_num_layers = cfg.enc_num_layers,
            dec_num_layers = cfg.dec_num_layers,
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout,
            encoder_pe = "2D",
            decoder_pe = "circular",
            decoder_lut = "memory",
        )
        self.automatic_optimization = False
        
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        
        self.set_cfg(cfg)
        self.train_outputs = []
        
        self.val_optimal_tour_distances = []
        self.val_predicted_tour_distances = []
        
        self.test_optimal_tour_distances = []
        self.test_predicted_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr*len(self.cfg.gpus), betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

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
        ntokens = batch["ntokens"]
        norm_consts = batch["norm_consts"]
        opt_tours = batch["opt_tours"]
        
        batch_size = src.shape[0]
        node_size = src.shape[1]
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
        
        predicted_tour_distance = self.get_tour_distance(src_original, ys) * norm_consts
        optimal_tour_distance = self.get_tour_distance(src_original, opt_tours) * norm_consts
        
        result = {
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            "optimal_tour_distance": optimal_tour_distance.tolist()
            }    
        
        self.test_predicted_tour_distances.extend(result["predicted_tour_distance"])
        self.test_optimal_tour_distances.extend(result["optimal_tour_distance"])
                
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
    
    def on_test_epoch_end(self):
        local_optimal = sum(self.test_optimal_tour_distances)
        local_predicted = sum(self.test_predicted_tour_distances)
        
        self.test_optimal_tour_distances.clear()
        self.test_predicted_tour_distances.clear()
        
        optimal_tour_distances = self.all_gather(local_optimal)
        predicted_tour_distances = self.all_gather(local_predicted)
        
        if self.trainer.is_global_zero:
            opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
            mean_opt_gap = opt_gaps.mean().item() * 100
            
            self.pred = predicted_tour_distances
            self.gt = optimal_tour_distances
            self.opt_gap = mean_opt_gap
            
            self.print(
                f"\npredicted_tour_distances={predicted_tour_distances}",
                f"\noptimal_tour_distances={optimal_tour_distances}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
            )

def parse_arguments(file_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"./{file_path}", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    return args

def generate_config_file(config_dict, file_path):
    with open(file_path, 'w') as f:
        for key, value in config_dict.items():
            if isinstance(value, list):
                value_str = '[' + ', '.join(str(item) for item in value) + ']'
            else:
                value_str = str(value)
            f.write(f"{key}: {value_str}\n")

if __name__ == "__main__":
    config_dict_template = {
        'train_data_path': './tsp100_train_concorde.txt',
        'val_data_path': None,
        'node_size': None,
        'train_batch_size': 80,
        'val_batch_size': 128,
        'test_batch_size': 1,
        'G': None,
        'resume_checkpoint': None,
        'gpus': [0],
        'max_epochs': 100,
        'enc_num_layers': None,
        'dec_num_layers': None,
        'd_model': None,
        'd_ff': None,
        'h': 8,
        'dropout': 0.1,
        'smoothing': 0.1,
        'seed': 1,
        'lr': 0.5,
        'betas': [0.9, 0.98],
        'eps': 1e-9,
        'factor': 1.0,
        'warmup': 400,
        'encoder_pe': '2D', # None, 2D
        'decoder_pe': 'circular PE', # None, 1D, circular 
        'decoder_lut': 'memory', # shared, unshared, memory
        'comparison_matrix': 'memory' # encoder_lut, decoder_lut, memory
    }
    
    data = []

    for file_path in os.listdir("./tsplib"):
        if "opt" in file_path:
            tsplib_path = f"{file_path.split('.')[0]}.tsp"
            
            if tsplib_path not in os.listdir("./tsplib"):
                continue
                
            problem = tsplib95.load(os.path.join("./tsplib", tsplib_path))
            coor_len = len(problem.node_coords.items())
            if coor_len == 0:
                continue
                
            if problem.dimension <= 500:
                name = problem.name.split(".")[0]
                print(name)

                config_dict = config_dict_template.copy()

                config_dict['val_data_path'] = f"./tsplib/{name}.tsp"
                config_dict['node_size'] = problem.dimension
                config_dict['G'] = problem.dimension
                
                for idx, checkpoint in enumerate([
                    "./TSP100-epoch=79-opt_gap=1.8782.ckpt", 
                    "./TSP100-epoch=55-opt_gap=0.4280.ckpt", 
                    "./TSP500-epoch=96-opt_gap=17.7910.ckpt", 
                    "./TSP500-epoch=98-opt_gap=6.5113.ckpt"
                ]):
                    if idx % 2 == 0:
                        config_dict['enc_num_layers'] = 6
                        config_dict['dec_num_layers'] = 6
                        config_dict['d_model'] = 128
                        config_dict['d_ff'] = 512
                    else:
                        config_dict['enc_num_layers'] = 12
                        config_dict['dec_num_layers'] = 12
                        config_dict['d_model'] = 256
                        config_dict['d_ff'] = 1024
                        
                    config_dict['resume_checkpoint'] = checkpoint
                    
                    file_path = f"{name}_{idx}.yaml"
                    generate_config_file(config_dict, file_path)
                    
                    args = parse_arguments(file_path)
                    cfg = OmegaConf.load(args.config)
                    pl.seed_everything(cfg.seed)
                    
                    tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint)
                    tsp_model.model.decoder_pe = PositionalEncoding_Circular(cfg.d_model, cfg.dropout, cfg.node_size)
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
                    
                    data.append([name, checkpoint, tsp_model.pred.item(), tsp_model.gt.item(), tsp_model.opt_gap])
                
                    if os.path.exists(file_path):
                        os.remove(file_path)

    csv_file_path = "output.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['problem_name', 'checkpoint', 'prediction', 'ground_truth', 'opt_gap'])
        writer.writerows(sorted(data, key = lambda x: float(x['problem_name'])))
    print(f"데이터를 {csv_file_path}에 성공적으로 저장했습니다.")
