import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

class SiameseTensorPairDataset(Dataset):
    def __init__(self, label_dir, tensor_dir, shuffle_pairs=True, val=None):
        '''
        Create an iterable dataset from a directory containing precomputed tensors of coin images and 
        a CSV file mapping tensor filenames to die IDs.
        
        Parameters:
                label_dir (str):          Path to the CSV file containing the tensor filename-die ID mapping.
                tensor_dir (str):         Path to directory containing the tensors.
                shuffle_pairs (boolean):  True for training (random pairs), False for testing (deterministic pairs).
        Returns:
                A tuple ((tensor, tensor_pair), label) where the first element is a tuple containing the image pairs 
                (tensors) and the second element is a Pytorch tensor indicating whether the pair
                is positive (1) or negative (0).
        '''
        self.tensor_df = pd.read_csv(label_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs
        self.val = val

    def __len__(self):
        return len(self.tensor_df)
    
    def __getitem__(self, idx):
        if self.val:
            random.seed(42)
        # Construct the path for the first tensor
        tensor_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[idx, 0])
        tensor = torch.load(tensor_path, map_location='cpu')
        label = self.tensor_df.iloc[idx, 1]

        # Randomly choose to get a positive or negative pair
        positive_pair = random.choice([True, False])

        if positive_pair:
            # Find another tensor of the same class
            same_class_indices = self.tensor_df[self.tensor_df[1] == label].index.tolist()

            if len(same_class_indices) == 1: # If only one tensor in this class, pair it with itself
                pair_idx = idx
            else:
                same_class_indices.remove(idx) # Exclude the current tensor
                pair_idx = random.choice(same_class_indices)
        else:
            # Find another tensor of a different class
            different_class_indices = self.tensor_df[self.tensor_df[1] != label].index.tolist()
            pair_idx = random.choice(different_class_indices)

        # Load the paired tensor
        tensor_pair_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[pair_idx, 0])
        tensor_pair = torch.load(tensor_pair_path, map_location='cpu')

        tensor = torch.squeeze(tensor)
        tensor_pair = torch.squeeze(tensor_pair)

        # Return the tensor pair and the label (1 for positive pair, 0 for negative pair)
        return (tensor, tensor_pair), torch.tensor([int(positive_pair)], dtype=torch.float32)