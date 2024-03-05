import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

class SiameseTripletTensorDataset(Dataset):
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
        anchor_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[idx, 0])
        anchor_tensor = torch.load(anchor_path, map_location='cpu')
        label = self.tensor_df.iloc[idx, 1]

        # Get indices of all tensors within the chosen class
        same_class_indices = self.tensor_df[self.tensor_df[1] == label].index.tolist()

        if len(same_class_indices) == 1: # If only one tensor in this class, positive tensor is itself
            positive_tensor = torch.load(anchor_path, map_location='cpu')

        else:
            same_class_indices.remove(idx) # Exclude the anchor tensor index
            positive_idx = random.choice(same_class_indices)
            positive_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[positive_idx, 0])
            positive_tensor = torch.load(positive_path, map_location='cpu')

        different_class_indices = self.tensor_df[self.tensor_df[1] != label].index.tolist()
        negative_idx = random.choice(different_class_indices)
        negative_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[negative_idx, 0])
        negative_tensor = torch.load(negative_path, map_location='cpu')
        
        anchor_tensor, positive_tensor, negative_tensor = map(lambda x: x.squeeze(), [anchor_tensor, positive_tensor, negative_tensor])
        # Return the anchor, positive and negative tensors
        return anchor_tensor, positive_tensor, negative_tensor