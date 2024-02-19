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

def cur_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    uk_time = timezone('Europe/London')
    loc_dt = datetime.now(uk_time)
    return loc_dt.strftime(fmt).replace(' ', '_')

def plot_loss(train_losses, val_losses, out_path):
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss_plot.jpg'))
    plt.close()

def plot_accuracy(train_accs, val_accs, out_path):
    plt.plot(train_accs, label='train acc')
    plt.plot(val_accs, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'accuracy_plot.jpg'))
    plt.close()

def train_siamese():
    # Add ArgumentParser() later on

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:7'
    print("Device: ", device)
    # Directory Config
    train_dir = config.obverse_train_dir
    train_csv = config.obverse_train_csv
    val_dir = config.obverse_validate_dir
    val_csv = config.obverse_validate_csv

    runs_dir = config.output_path

    # Create Directory to Store Experiment Artifacts
    artifact_dir_name = cur_time()
    artifact_path = os.path.join(runs_dir, artifact_dir_name)
    os.makedirs(artifact_path)

    # Hyperparameters
    batch_size = 1
    num_epochs = 20
    learning_rate = 1e-5

    # Training Settings - later to be implemented with ArgumentParser()
    contra_loss = None
    contra_margin = 1

    # ViT_b_16 transform
    transform = ViT_B_16_Weights.DEFAULT.transforms()

    train_dataset = SiamesePairDataset(label_dir=train_csv, tensor_dir=train_dir, transform=transform)
    val_dataset = SiamesePairDataset(label_dir=val_csv, tensor_dir=val_dir, transform=transform)

    # Default batch_size = 1 if not set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model for pre-computed SAM feature vectors
    # model = SiameseNetworkSAM(contrastive_loss=contra_loss)
    # model.to(device)

    # Pretrained Torch models
    # This iteration - vit_b_16 trained on ImageNet1k
    model = SiameseNetwork()
    model.to(device)

    # Initialise Loss Function
    if contra_loss:
        criterion = ContrastiveLoss(margin=contra_margin)
    else:
        criterion = nn.BCELoss()

    # Initialise Optimizer - can experiment with different optimizers
    # Here we use Adam  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    optimizer.zero_grad()

    best_val_loss = 100000000

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print("Epoch [{} / {}]".format(epoch+1, num_epochs))
        epoch_start = time.time()
        model.train()

        losses = []
        
        correct_pred = 0
        total_pred = 0

        # Training Loop Start
        for i, ((positive_pairs, pos_label), (negative_pairs, neg_label)) in enumerate(train_dataloader):
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

        # scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = correct_pred / total_pred

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        print("Training: Loss={:.2f} | Accuracy={:.2f}".format(sum(train_losses)/len(train_losses), correct_pred / total_pred))
        # Training Loop End

        # Validation Loop Start
        model.eval()

        losses = []
        
        correct_pred = 0
        total_pred = 0

        for i, ((positive_pairs, pos_label), (negative_pairs, neg_label)) in enumerate(val_dataloader):
            pos1, pos2 = positive_pairs
            neg1, neg2 = negative_pairs
            pos1, pos2, neg1, neg2, pos_label, neg_label = map(lambda x: x.to(device), [pos1, pos2, neg1, neg2, pos_label, neg_label])

            with torch.no_grad():
                pos_prob = model(pos1, pos2)
                pos_loss = criterion(pos_prob, pos_label)
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
            
        val_loss = sum(losses) / max(1, len(losses))
        avg_val_acc = correct_pred / total_pred

        val_losses.append(val_loss)
        val_accs.append(avg_val_acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print("Validation: Loss={:.2f} | Accuracy={:.2f} | Time={:.2f}".format(val_loss, correct_pred / total_pred, epoch_time))
        # Validation Loop End

        # Update "best.pt" model if val_loss of current epoch is lower than the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict()
                },
                os.path.join(artifact_path, "best.pt")
            )

        # Plot Loss and Accuracy 
        plot_loss(train_losses, val_losses, artifact_path)
        plot_accuracy(train_accs, val_accs, artifact_path)


if __name__ == "__main__":
    train_siamese()