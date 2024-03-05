import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SiameseTensorPairDataset(Dataset):
    def __init__(self, label_dir, tensor_dir):
        self.tensor_df = pd.read_csv(label_dir, header=None)
        self.tensor_dir = tensor_dir

        # Generate all possible positive pairs
        self.positive_pairs = []
        self.negative_pairs = []

        die_ids = self.tensor_df[1].unique()
        group_indices = {die_id: self.tensor_df.index[self.tensor_df[1] == die_id].tolist() for die_id in die_ids}

        # Create all possible positive pairs within each class
        for indices in group_indices.values():
            if len(indices) > 1:  # Only if there are at least 2 samples in the class
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        self.positive_pairs.append((indices[i], indices[j]))
        
        # Number of needed negative pairs is the same as positive pairs
        num_negatives = len(self.positive_pairs)

        # Create negative pairs
        while len(self.negative_pairs) < num_negatives:
            idx1, idx2 = np.random.choice(len(self.tensor_df), 2, replace=False)
            # Ensure they are from different classes
            if self.tensor_df.iloc[idx1, 1] != self.tensor_df.iloc[idx2, 1]:
                self.negative_pairs.append((idx1, idx2))

        print(len(self.positive_pairs))
        print(len(self.negative_pairs))

    def __len__(self):
        # The dataset length is now the sum of positive and negative pairs
        return len(self.positive_pairs) + len(self.negative_pairs)
    
    def __getitem__(self, idx):
        # Determine if the index is for a positive or negative pair
        if idx < len(self.positive_pairs):
            # It's a positive pair
            idx1, idx2 = self.positive_pairs[idx]
            label = torch.tensor([1], dtype=torch.float32)  # Positive label
        else:
            # It's a negative pair
            # Adjust index because idx starts from 0 for negative pairs in this logic
            adjusted_idx = idx - len(self.positive_pairs)
            idx1, idx2 = self.negative_pairs[adjusted_idx]
            label = torch.tensor([0], dtype=torch.float32)  # Negative label

        # Construct the path and load tensors for the pair
        tensor_path1 = os.path.join(self.tensor_dir, self.tensor_df.iloc[idx1, 0])
        tensor_path2 = os.path.join(self.tensor_dir, self.tensor_df.iloc[idx2, 0])
        tensor1 = torch.load(tensor_path1, map_location='cpu')
        tensor2 = torch.load(tensor_path2, map_location='cpu')

        # Ensure tensors are not batched
        tensor1 = torch.squeeze(tensor1)
        tensor2 = torch.squeeze(tensor2)

        # Return the tensor pair and the label
        return (tensor1, tensor2), label
