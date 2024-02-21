import os
import time
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

import config
from tensorsiamese import SiameseNetworkSAM
from contrastive import ContrastiveLoss
from tensorpair_v1 import SiameseTensorPairDataset
from imgpair_v1 import SiamesePairDataset
from siamese import SiameseNetwork
# from siamese_example import SiameseNetwork

from torchvision.models import ViT_B_16_Weights

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


def train_samsiamese():
    # Add ArgumentParser() later on

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda'
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
    num_epochs = 100
    learning_rate = 1e-4

    # Training Settings - later to be implemented with ArgumentParser()
    contra_margin = 1

    # transform = ViT_B_16_Weights.DEFAULT.transforms()
    # transform = transforms.Compose([
    #     transforms.Resize((512,512)),
    #     transforms.ToTensor(),
    # ])

    train_dataset = SiameseTensorPairDataset(label_dir=train_csv, tensor_dir=train_dir)
    val_dataset = SiameseTensorPairDataset(label_dir=val_csv, tensor_dir=val_dir)

    # # Default batch_size = 1 if not set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model for pre-computed SAM feature vectors
    # model = SiameseNetworkSAM(contrastive_loss=contra_loss)
    # model.to(device)

    # Pretrained Torch models
    # This iteration - vit_b_16 trained on ImageNet1k
    model = SiameseNetworkSAM()
    model.to(device)

    # Initialise Loss Function
    # criterion = nn.BCELoss()
    criterion = nn.BCELoss()

    # Initialise Optimizer - can experiment with different optimizers
    # Here we use Adam  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_val_loss = 100000000

    epoch_train_losses = []
    epoch_val_losses = []
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
        for i, ((tensor1, tensor2), label) in enumerate(train_dataloader):
            tensor1, tensor2, label = map(lambda x: x.to(device), [tensor1, tensor2, label])
            label = label.view(-1)

            optimizer.zero_grad()
            prob = model(tensor1, tensor2)
            loss = criterion(prob, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct_pred += torch.count_nonzero(label == (prob > 0.5)).item()
            total_pred += len(label)

        # scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        epoch_train_acc = correct_pred / total_pred

        train_losses.append(avg_train_loss)
        epoch_train_loss = sum(train_losses)/len(train_losses)
        epoch_train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        print("Training: Loss={:.2f} | Accuracy={:.2f}".format(epoch_train_loss, epoch_train_acc))
        # Training Loop End

        # Validation Loop Start
        model.eval()

        losses = []
        accs = []
        
        correct_pred = 0
        total_pred = 0

        for (tensor1, tensor2), label in val_dataloader:
            tensor1, tensor2, label = map(lambda x: x.to(device), [tensor1, tensor2, label])
            label = label.view(-1)
            with torch.no_grad():
                prob = model(tensor1, tensor2)
                loss = criterion(prob, label)

            losses.append(loss.item())
            correct_pred += torch.count_nonzero(label == (prob > 0.5)).item()
            total_pred += len(label)
            
        val_loss = sum(losses) / max(1, len(losses))
        epoch_val_acc = correct_pred / total_pred

        val_losses.append(val_loss)
        val_accs.append(epoch_val_acc)

        epoch_val_loss = sum(val_losses)/len(val_losses)
        epoch_val_losses.append(epoch_val_loss)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print("Validation: Loss={:.2f} | Accuracy={:.2f} | Time={:.2f}".format(epoch_val_loss, epoch_val_acc, epoch_time))
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
        plot_loss(epoch_train_losses, epoch_val_losses, artifact_path)
        plot_accuracy(train_accs, val_accs, artifact_path)

if __name__ == "__main__":
    train_samsiamese()