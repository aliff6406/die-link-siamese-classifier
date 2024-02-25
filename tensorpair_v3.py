import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

class SiameseTensorPairDataset(Dataset):
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
        class_label = random.choice(unique_labels)

        # Get indices of all tensors within the chosen class
        class_indices = self.tensor_df[self.tensor_df[1] == class_label].index.tolist()

        # Randomly select 3 tensors from this class
        if len(class_indices) >= 3:
            selected_indices = random.sample(class_indices, 3)
        else:
            selected_indices = random.choices(class_indices, k=3)  # Allow repeats if less than 3

        # Load the tensors and squeeze them
        selected_tensors = [torch.load(os.path.join(self.tensor_dir, self.tensor_df.iloc[i, 0]), map_location=self.device) for i in selected_indices]

        # Choose indices for the positive pair
        positive_indices = random.sample(range(len(selected_tensors)), 2)
        positive_pair_tensors = [selected_tensors[i] for i in positive_indices]

        # Find the index for the negative pair tensor (the one not used in the positive pair)
        negative_index = list(set(range(len(selected_tensors))) - set(positive_indices))[0]
        negative_pair_tensor = selected_tensors[negative_index]

        # Find a different class, ensuring it's not the same as the current class
        different_labels = [label for label in unique_labels if label != class_label]
        different_class_label = random.choice(different_labels)

        # Get all indices of tensors belonging to the different class and randomly select one
        different_class_indices = self.tensor_df[self.tensor_df[1] == different_class_label].index.tolist()
        different_class_tensor_index = random.choice(different_class_indices)
        different_class_tensor = torch.load(os.path.join(self.tensor_dir, self.tensor_df.iloc[different_class_tensor_index, 0]), map_location='cpu')

        # Return positive and negative pairs with their labels (1 for positive, 0 for negative)
        pos_tensor1 = positive_pair_tensors[0]
        pos_tensor2 = positive_pair_tensors[1]
        neg_tensor1 = negative_pair_tensor
        neg_tensor2 = different_class_tensor
        return pos_tensor1, pos_tensor2, torch.tensor([1], dtype=torch.float32), neg_tensor1, neg_tensor2, torch.tensor([0], dtype=torch.float32)