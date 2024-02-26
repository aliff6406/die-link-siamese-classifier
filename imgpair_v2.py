import os
import random

import numpy as np
import pandas as pd

import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class SiamesePairDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir, img_dir, shuffle_pairs=True, transform=None, augment=None):
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
        class_label = self.image_df.iloc[idx, 1]
    
        # Find all images of this class
        class_indices = self.image_df[self.image_df[1] == class_label].index.tolist()
    

        # Randomly select 3 images from this class
        selected_indices = random.sample(class_indices, 3)
        selected_images = [Image.open(os.path.join(self.img_dir, self.image_df.iloc[i, 0])) for i in selected_indices]

        # Randomly select one image for the positive pair and one for the negative pair
        positive_pair_images = random.sample(selected_images, 2)
        negative_pair_image = [image for image in selected_images if image not in positive_pair_images][0]
        
        # Find a different class, ensuring it's not the same as the current class
        different_labels = list(self.image_df[self.image_df[1] != class_label][1].unique())
        different_class_label = random.choice(different_labels) 
        
        # Get all indices of images belonging to the different class and randomly select one
        different_class_indices = self.image_df[self.image_df[1] == different_class_label].index.tolist()
        different_class_image_index = random.choice(different_class_indices)
        different_class_image = Image.open(os.path.join(self.img_dir, self.image_df.iloc[different_class_image_index, 0]))
        
        # Apply transform
        if self.transform:
            positive_pair_images = [self.transform(img) for img in positive_pair_images]
            negative_pair_image = self.transform(negative_pair_image)
            different_class_image = self.transform(different_class_image)
        
        # Return positive and negative pairs with their labels (1 for positive, 0 for negative)
        positive_pair = (positive_pair_images[0], positive_pair_images[1])
        negative_pair = (negative_pair_image, different_class_image)
        return (positive_pair, torch.tensor([1], dtype=torch.float32)), (negative_pair, torch.tensor([0], dtype=torch.float32))
