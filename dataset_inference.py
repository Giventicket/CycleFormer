import torch
from torch.utils.data import DataLoader, Dataset
from model import subsequent_mask

from tqdm import tqdm
from pprint import pprint

"""
   src
   tgt
   src_mask
   tgt_mask
   tgt_y
   ntokens
   
   blank => -1
"""


class TSPDataset(Dataset):
    def __init__(
        self,
        data_path=None,
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.tsp_instances = []
        self.tsp_tours = []

        self._readDataFile()
        
        self.raw_data_size = len(self.tsp_instances)
        self.max_node_size = len(self.tsp_tours[0])
        
        self.src = []
        self.tgt = []
        self.visited_mask = []
        self.tgt_y = []
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
        """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
        """
        with open(self.data_path, "r") as fp:
            tsp_set = fp.readlines()
            for idx, tsp in enumerate(tsp_set):
                    
                tsp = tsp.split("output")
                tsp_instance = tsp[0].split()

                tsp_instance = [float(i) for i in tsp_instance]
                loc_x = torch.FloatTensor(tsp_instance[::2])
                loc_y = torch.FloatTensor(tsp_instance[1::2])
                tsp_instance = torch.stack([loc_x, loc_y], dim=1)
                self.tsp_instances.append(tsp_instance)

                tsp_tour = tsp[1].split()
                tsp_tour = [(int(i) - 1) for i in tsp_tour]
                
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                self.tsp_tours.append(tsp_tour)
        return

    def _process(self):
        for tsp_instance, tsp_tour in tqdm(zip(self.tsp_instances, self.tsp_tours)):
            ntoken = 1
            self.ntokens.append(torch.LongTensor([ntoken]))
            self.src.append(tsp_instance)
            self.tgt.append(tsp_tour[0:ntoken])
            
            visited_mask = torch.zeros(ntoken, self.max_node_size, dtype=torch.bool)
            for v in range(ntoken):
                visited_mask[v: , self.tgt[-1][v]] = True # visited
                
            self.visited_mask.append(visited_mask)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.visited_mask[idx], self.ntokens[idx], self.tsp_tours[idx]


def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    src = [ele[0] for ele in batch]
    tgt = [ele[1] for ele in batch]
    visited_mask = [ele[2] for ele in batch]
    ntokens = [ele[3] for ele in batch]
    tsp_tours = [ele[4] for ele in batch]

    tgt = torch.stack(tgt, dim=0)
    
    return {
        "src": torch.stack(src, dim=0),
        "tgt": tgt,
        "visited_mask": torch.stack(visited_mask, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "tgt_mask": make_tgt_mask(tgt),
        "tsp_tours": torch.stack(tsp_tours, dim=0),
    }


if __name__ == "__main__":
    train_dataset = TSPDataset("./tsp20_test_concorde.txt")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        for k, v in tsp_instances.items():
            print(k, v.shape)
        print()
        break
