import os
import random

import numpy as np
import pandas as pd

import torch
from PIL import Image

# For testing
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def visualize_dataloader_pairs(dataloader):
    """
    Visualizes the first batch of positive and negative image pairs from a DataLoader.

    Parameters:
    - dataloader: A DataLoader instance wrapping a SiamesePairDataset.
    """
    # Get the first batch from the dataloader
    batch = next(iter(dataloader))
    
    # Unpack the batch
    positive_pairs, positive_labels, negative_pairs, negative_labels = batch
    
    # Determine the number of pairs to visualize
    batch_size = len(positive_labels)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(batch_size, 4, figsize=(20, 5 * batch_size))  # Adjust subplot size as needed
    
    for i in range(batch_size):
        # Images
        pos_img1, pos_img2 = positive_pairs[i]
        neg_img1, neg_img2 = negative_pairs[i]
        
        # Titles for the subplots
        titles = ['Positive Pair - Image 1', 'Positive Pair - Image 2', 'Negative Pair - Image 1', 'Negative Pair - Image 2']
        
        # Flatten the axis array for easy indexing if in 2D (for batch_size > 1)
        if batch_size > 1:
            axs_flat = axs[i].ravel()
        else:
            axs_flat = axs.ravel()
        
        # Display each image in the pair
        for j, img in enumerate([pos_img1, pos_img2, neg_img1, neg_img2]):
            # if img.dim() == 3:  # If the image is a tensor, convert it to a PIL Image for visualization
                # img = transforms.ToPILImage()(img)
            axs_flat[j].imshow(img)
            axs_flat[j].set_title(titles[j])
            axs_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()
# Example usage

class SiamesePairDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir, img_dir, shuffle_pairs=True, transform=None):
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

if __name__ == "__main__":
    dataset = SiamesePairDataset(label_dir='./data/ccc_images_final/train/obverse_train_labels.csv', img_dir='./data/ccc_images_final/train/obverses/')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Adjust batch_size as needed

    visualize_dataloader_pairs(dataloader)
