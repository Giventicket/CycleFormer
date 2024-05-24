import torch
from torch.utils.data import DataLoader, Dataset
from model import subsequent_mask

from tqdm import tqdm
from pprint import pprint

import os
import tsplib95

class TSPDataset(Dataset):
    def __init__(
        self,
        data_path="./tsplib/pr107.tsp",
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.tsp_instances = []
        self.opt_tours = []
        self.norm_consts = []

        self._readDataFile()
        
        self.raw_data_size = len(self.tsp_instances)
        self.max_node_size = len(self.tsp_instances[0])
        
        self.src = []
        self.ntokens = []
        self._process()
        self.data_size = len(self.src)

        print()
        print("#### processing dataset... ####")
        print("data_path:", data_path)
        print("raw_data_size:", self.raw_data_size)
        print("max_node_size:", self.max_node_size)
        print("data_size:", self.data_size)
        print()

    def _readDataFile(self):
        problem = tsplib95.load(self.data_path)
        opt_tour = tsplib95.load(self.data_path.split(".tsp")[0] + ".opt.tour")
        
        xs = []
        ys = []
        
        for idx, (x, y) in problem.node_coords.items():
            xs.append(x)
            ys.append(y)
            
        loc_x = torch.FloatTensor(xs)
        loc_y = torch.FloatTensor(ys)
        
        norm_const = torch.max(torch.cat([loc_x, loc_y]))
        tsp_instance = torch.stack([loc_x / norm_const, loc_y / norm_const], dim=1)
        self.tsp_instances.append(tsp_instance)
        self.norm_consts.append(norm_const)
        opt_tour = torch.tensor(opt_tour.tours)[0] - 1
        self.opt_tours.append(opt_tour)
        
        return
    
    def _process(self):
        for tsp_instance in tqdm(self.tsp_instances):
            ntoken = 1
            self.ntokens.append(torch.LongTensor([ntoken]))
            self.src.append(tsp_instance)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.ntokens[idx], self.norm_consts[idx], self.opt_tours[idx]


def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    src = [ele[0] for ele in batch]
    ntokens = [ele[1] for ele in batch]
    norm_consts = [ele[2] for ele in batch]
    opt_tours = [ele[3] for ele in batch]
    
    return {
        "src": torch.stack(src, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "norm_consts": torch.stack(norm_consts, dim=0),
        "opt_tours": torch.stack(opt_tours, dim=0),
    }


if __name__ == "__main__":
    train_dataset = TSPDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        for k, v in tsp_instances.items():
            print(k, v)
        print()
        break
