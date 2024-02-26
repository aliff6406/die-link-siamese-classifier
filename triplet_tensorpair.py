import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

class SiameseTripletTensorDataset(Dataset):
    def __init__(self, label_dir, tensor_dir, shuffle_pairs=True):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.tensor_df)
    
    def __getitem__(self, idx):
        # Randomly select a class with equal probability
        unique_labels = self.tensor_df[1].unique()
        label = random.choice(unique_labels)

        # Get indices of all tensors within the chosen class
        same_label_indices = self.tensor_df[self.tensor_df[1] == label].index.tolist()

        # Randomly select a tensor
        idx = random.choice(same_label_indices)
        tensor_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[idx, 0])
        anchor_tensor = torch.load(tensor_path, map_location='cpu')

        # Randomly choose to get a positive or negative pair
        positive_pair = random.choice([True, False])

        # Find another tensor of the same class
        same_label_indices = self.tensor_df[self.tensor_df[1] == label].index.tolist()

        if len(same_label_indices) == 1: # If only one tensor in this class, pair it with itself
            pair_idx = idx
        else:
            same_label_indices.remove(idx) # Exclude the current tensor
            positive_pair_idx = random.choice(same_label_indices)
        
        different_labels = [label_ for label_ in unique_labels if label_ != label]
        different_label = random.choice(different_labels)
        different_label_indices = self.tensor_df[self.tensor_df[1] == different_label].index.tolist()
        negative_pair_idx = random.choice(different_label_indices)

        # Load the paired tensor
        positive_tensor_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[positive_pair_idx, 0])
        positive_tensor = torch.load(positive_tensor_path, map_location='cpu')

        negative_tensor_path = os.path.join(self.tensor_dir, self.tensor_df.iloc[negative_pair_idx, 0])
        negative_tensor = torch.load(negative_tensor_path, map_location='cpu')

        anchor_tensor, positive_tensor, negative_tensor = map(lambda x: x.squeeze(), [anchor_tensor, positive_tensor, negative_tensor])
        # Return the anchor, positive and negative tensors
        return anchor_tensor, positive_tensor, negative_tensor