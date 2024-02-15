import os
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import config
from samsiamese import SiameseNetworkSAM
from contrastive import ContrastiveLoss
from src.data.siamese_pair_dataset import SiamesePairDataset

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directory Config
    sam_checkpoint = config.sam_checkpoint
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
    learning_rate = 0.0005

    # Training Settings - later to be implemented with ArgumentParser()
    contra_loss = None
    contra_margin = 1

    sam_transform = v2.Compose([
        v2.ToImage(), # Convert to tensor from PIL image
        v2.Resize(size=(1024,1024), antialias=False) # Resize to 1024x1024
    ])

    train_dataset = SiamesePairDataset(label_dir=train_csv, img_dir=train_dir, transform=sam_transform)
    val_dataset = SiamesePairDataset(label_dir=val_csv, img_dir=val_dir, transform=sam_transform)

    # Default batch_size = 1 if not set
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    samsiamese_model = SiameseNetworkSAM(device=device, checkpoint=sam_checkpoint, contrastive_loss=contra_loss)
    samsiamese_model.to(device)

    # Initialise Loss Function
    if contra_loss:
        criterion = ContrastiveLoss(margin=contra_margin)
    else:
        criterion = nn.BCELoss()

    # Initialise Optimizer - can experiment with different optimizers
    # Here we use Adam  
    optimizer = torch.optim.Adam(samsiamese_model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=artifact_path)

    best_val_loss = 100000000

    for epoch in range(num_epochs):
        print("[{} / {}]".format(epoch+1, num_epochs))
        samsiamese_model.train()

        losses = []
        train_losses = []
        train_accs = []
        correct_pred = 0
        total_pred = 0

        # Training Loop Start
        for (img1, img2), label, (class1, class2) in train_dataloader:
            img1, img2, label = map(lambda x: x.to(device), [img1, img2, label])

            optimizer.zero_grad()
            if contra_loss:
                output1, output2 = samsiamese_model(img1, img2)
                loss = criterion(output1, output2, label)
                loss.backwards()
                optimizer.step()

                losses.append(loss.item())
                correct_pred += torch.count_nonzero(label == (F.pairwise_distance(output1, output2) < contra_margin)).item()
                total_pred += len(label)
            else:
                prob = samsiamese_model(img1, img2)
                loss = criterion(prob, label)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                correct_pred += torch.count_nonzero(label == (prob > 0.5)).item()
                total_pred += len(label)

        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = correct_pred / total_pred

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        writer.add_scalar("train_loss", avg_train_loss, epoch+1)
        writer.add_scalar("train_accuracy", avg_train_acc, epoch+1)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(train_losses)/len(train_losses), correct_pred / total_pred))
        # Training Loop End

        # Validation Loop Start
        samsiamese_model.eval()

        losses = []
        val_losses = []
        val_accs = []
        correct_pred = 0
        total_pred = 0

        for (img1, img2), label, (class1, class2) in val_dataloader:
            img1, img2, label = map(lambda x: x.to(device), [img1, img2, label])

            if contra_loss:
                output1, output2 = samsiamese_model(img1, img2)
                loss = criterion(output1, output2, label)

                losses.append(loss.item())
                correct_pred += torch.count_nonzero(label == (F.pairwise_distance(output1, output2) < contra_margin)).item()
                total_pred += len(label)
            else:
                prob = samsiamese_model(img1, img2)
                loss = criterion(prob, label)

                losses.append(loss.item())
                correct_pred += torch.count_nonzero(label == (prob > 0.5)).item()
                total_pred += len(label)
            
        val_loss = sum(losses) / max(1, len(losses))
        avg_val_acc = correct_pred / total_pred

        val_losses.append(val_loss)
        val_accs.append(avg_val_acc)

        writer.add_scalar("val_loss", val_loss, epoch+1)
        writer.add_scalar("val_acc", correct_pred / total_pred, epoch+1)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct_pred / total_pred))
        # Validation Loop End

        # Update "best.pt" model if val_loss of current epoch is lower than the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": samsiamese_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(artifact_path, "best.pt")
            )
    
    # Plot Loss and Accuracy 
    plot_loss(train_losses, val_losses, artifact_path)
    plot_accuracy(train_accs, val_accs, artifact_path)

if __name__ == "__main__":
    train_samsiamese()