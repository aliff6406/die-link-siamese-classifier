import os
import time
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models import ViT_B_16_Weights

import matplotlib.pyplot as plt

import config
from tensorsiamese import SiameseNetworkSAM
from siamese import SiameseNetwork
from contrastive import ContrastiveLoss
from imgpair_v2 import SiamesePairDataset


if __name__ == "__main__":
    device = 'mps'

    learning_rate = 1e-5

    # ViT_b_16 transform
    transform = ViT_B_16_Weights.DEFAULT.transforms()

    # Instantiate the dataset
    dataset = SiamesePairDataset(
        label_dir='./data/ccc_images_final/train/obverse_train_labels.csv',
        img_dir='./data/ccc_images_final/train/obverses/',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SiameseNetwork()
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    losses = []
    correct_pred = 0
    total_pred = 0
    for i, ((positive_pairs, pos_label), (negative_pairs, neg_label)) in enumerate(dataloader):
        pos1, pos2 = positive_pairs
        neg1, neg2 = negative_pairs
        pos1, pos2, neg1, neg2, pos_label, neg_label = map(lambda x: x.to(device), [pos1, pos2, neg1, neg2, pos_label, neg_label])

        optimizer.zero_grad()
        pos_prob = model(pos1, pos2)
        pos_loss = criterion(pos_prob, pos_label)
        pos_loss.backward()  # Backpropagate loss for positive pairs
        optimizer.step()  # Update model parameters based on positive pairs
        losses.append(pos_loss.item())
        correct_pred += torch.count_nonzero(pos_label == (pos_prob > 0.5)).item()
        total_pred += len(pos_label)

        # Process negative pairs
        optimizer.zero_grad()  # Clear gradients before processing the negative pairs
        neg_prob = model(neg1, neg2)
        neg_loss = criterion(neg_prob, neg_label)
        neg_loss.backward()  # Backpropagate loss for negative pairs
        optimizer.step()  # Update model parameters based on negative pairs
        losses.append(neg_loss.item())
        correct_pred += torch.count_nonzero(neg_label == (neg_prob > 0.5)).item()
        total_pred += len(neg_label)

    avg_train_loss = sum(losses) / len(losses)
    avg_train_acc = correct_pred / total_pred

    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    print("Training: Loss={:.2f} | Accuracy={:.2f}".format(sum(train_losses)/len(train_losses), correct_pred / total_pred))
        # # Unpack the positive and negative pairs
        # pos_img1, pos_img2 = positive_pair
        # neg_img1, neg_img2 = negative_pair

        # # Convert the images to PIL for visualization, assuming they are Tensors
        # pos_img1 = to_pil_image(pos_img1.squeeze(0))
        # pos_img2 = to_pil_image(pos_img2.squeeze(0))
        # neg_img1 = to_pil_image(neg_img1.squeeze(0))
        # neg_img2 = to_pil_image(neg_img2.squeeze(0))

        # # Visualize the positive pair
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(pos_img1)
        # plt.title("Positive Pair - Image 1")
        # plt.subplot(1, 2, 2)
        # plt.imshow(pos_img2)
        # plt.title("Positive Pair - Image 2")
        # plt.show()

        # # Visualize the negative pair
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(neg_img1)
        # plt.title("Negative Pair - Image 1")
        # plt.subplot(1, 2, 2)
        # plt.imshow(neg_img2)
        # plt.title("Negative Pair - Image 2")
        # plt.show()