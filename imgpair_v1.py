import os
import random

import numpy as np
import pandas as pd

import torch
from PIL import Image
from torchvision import transforms

class SiamesePairDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir, img_dir, sam_backbone=False, shuffle_pairs=True, transform=None, augment=None):
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
        self.augment = augment

        if self.augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3, hue=0.3)
            ])

    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, idx):
        # Randomly choose a class with equal probability
        unique_labels = self.image_df[1].unique().tolist()
        chosen_label = random.choice(unique_labels)
        
        # Get indices of all images within the chosen class
        same_class_indices = self.image_df[self.image_df[1] == chosen_label].index.tolist()

        # Randomly select an image from the chosen class
        idx = random.choice(same_class_indices)
        img_path = os.path.join(self.img_dir, self.image_df.iloc[idx, 0])
        image = Image.open(img_path)

        # Randomly choose to get a positive or negative pair
        positive_pair = random.choice([True, False])

        if positive_pair:
            # Handle case where there is only one image in the class
            if len(same_class_indices) > 1:
                same_class_indices.remove(idx)  # Remove the current image's index
            pair_idx = random.choice(same_class_indices)
        else:
            different_labels = [label_ for label_ in unique_labels if label_ != chosen_label]
            different_label = random.choice(different_labels)
            different_class_indices = self.image_df[self.image_df[1] == different_label].index.tolist()
            pair_idx = random.choice(different_class_indices)

        img_pair_path = os.path.join(self.img_dir, self.image_df.iloc[pair_idx, 0])
        image_pair = Image.open(img_pair_path)

        if self.transform:
            image = self.transform(image)
            image_pair = self.transform(image_pair)
        if self.augment:
            image = self.augmentation(image)
            image_pair = self.augmentation(image_pair)

        # Return the image pair and the label (1 for positive pair, 0 for negative pair)
        return image, image_pair, torch.tensor([int(positive_pair)], dtype=torch.float32)

