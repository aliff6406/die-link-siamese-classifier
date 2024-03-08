import os

import pandas as pd
import torch
from torch.utils.data import Dataset

class OfflinePairDataset(Dataset):
    def __init__(self, pair_dir, tensor_dir, shuffle_pairs=True):
        self.pair_df = pd.read_csv(pair_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        tensor1_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 0])
        tensor2_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 1])
        label = self.pair_df.iloc[idx, 2]

        tensor1 = torch.load(tensor1_path, map_location='cpu').squeeze()
        tensor2 = torch.load(tensor2_path, map_location='cpu').squeeze()

        return tensor1, tensor2, torch.tensor(label, dtype=torch.float32)
    
class OfflineTripletDataset(Dataset):
    def __init__(self, pair_dir, tensor_dir, shuffle_pairs=True):
        self.triplet_df = pd.read_csv(pair_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.triplet_df)
    
    def __getitem__(self, idx):
        anc_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 0])
        pos_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 1])
        neg_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 2])

        anchor = torch.load(anc_path, map_location='cpu').squeeze()
        positive = torch.load(pos_path, map_location='cpu').squeeze()
        negative = torch.load(neg_path, map_location='cpu').squeeze()

        return anchor, positive, negative
    