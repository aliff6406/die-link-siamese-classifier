import os
import random

import numpy as np
import pandas as pd

import torch
from PIL import Image

class SiamesePairDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir, img_dir, sam_backbone=False, shuffle_pairs=True, transform=None):
        '''
        Create an iterable dataset from a directory containing images of coins and an Excel file mapping coin names to die IDs.
        
        Parameters:
                label_dir (str):           Path to the CSV file containing the coin-die mapping.
                img_dir (str):             Path to directory containing the images.
                shuffle_pairs (boolean):   True for training (random pairs), False for testing (deterministic pairs).
                Transform (boolean):       True if dataset requires transform operation (all backbones except SAM).
        Returns:
                A tuple ((image, image_pair), label) where the first element is a tuple containing the image pairs 
                (np arrays read by opencv) and the second element is a Pytorch tensor indicating whether the pair
                is positive (1) or negative (0).
                
        '''
        self.image_df = pd.read_csv(label_dir, header=None)
        self.img_dir = img_dir
        self.sam_backbone  = sam_backbone
        self.shuffle_pairs = shuffle_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_df.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.image_df.iloc[idx, 1]

        # Randomly choose to get a positive or negative pair
        positive_pair = random.choice([True, False])

        if positive_pair:
            # Find another image of the same class
            same_class_indices = self.image_df[self.image_df[1] == label].index.tolist() # Get a list of indices of images with labels == current idx label

            if len(same_class_indices) == 1: # Handles edge case if there is only one image in this class
                 pair_idx = idx # Use the same image to pair to itself
            else:
                same_class_indices.remove(idx) # Remove the current image's index so that it isnt paired with itself
                pair_idx = random.choice(same_class_indices)

        else:
            # Find another image of a different class
            different_class_indices = self.image_df[self.image_df[1] != label].index.tolist()
            pair_idx = random.choice(different_class_indices)

        img_pair_path = os.path.join(self.img_dir, self.image_df.iloc[pair_idx, 0])
        image_pair = Image.open(img_pair_path)

        if self.transform:
            image = self.transform(image)
            image_pair = self.transform(image_pair)


        # Return the image pair and the label (1 for positive pair, 0 for negative pair)
        return (image, image_pair), torch.tensor([int(positive_pair)], dtype=torch.float32)

