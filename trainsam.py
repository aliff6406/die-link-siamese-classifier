import os
import time
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

import config
from samsiamese import SiameseNetworkSAM
from contrastive import ContrastiveLoss
from imgpair_v1 import SiamesePairDataset
from sam import SAMImageEncoderViT

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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'mps'
    print("Device: ", device)
    # Directory Config
    train_dir = './data/ccc_images_final/388-325/train/obverses/'
    train_csv = './data/ccc_images_final/388-325/train/train_labels.csv'
    val_dir = './data/ccc_images_final/388-325/val/obverses/'
    val_csv = './data/ccc_images_final/388-325/val/val_labels.csv'

    runs_dir = config.output_path

    # Create Directory to Store Experiment Artifacts
    artifact_dir_name = cur_time()
    artifact_path = os.path.join(runs_dir, artifact_dir_name)
    os.makedirs(artifact_path)

    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    # base_lr = 5e-2
    # Linear Scaling of learning rate based on [https://arxiv.org/pdf/1706.02677.pdf]
    # learning_rate = base_lr * batch_size/256
    learning_rate = 5e-2
    # weight_decay = 5e-4

    # Training Settings - later to be implemented with ArgumentParser()
    contra_loss = None
    contra_margin = 1
    cosine = None

    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR, antialias=False),
        transforms.ToTensor()
    ])

    train_dataset = SiamesePairDataset(label_dir=train_csv, img_dir=train_dir, transform=transform, augment=True)
    val_dataset = SiamesePairDataset(label_dir=val_csv, img_dir=val_dir, transform=transform)

    # # Default batch_size = 1 if not set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Pretrained Torch models
    model = SiameseNetworkSAM(contrastive_loss=contra_loss)
    sam = SAMImageEncoderViT()
    model.to(device)
    sam.to(device)

    # Initialise Loss Function
    # criterion = nn.BCELoss()
    if contra_loss:
        criterion = ContrastiveLoss(margin=contra_margin)
    elif cosine:
        criterion = CosineEmbeddingLoss()
    else:
        criterion = nn.BCELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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

        train_loss = 0.0
        val_loss = 0.0
        correct_pred = 0
        total_pred = 0

        # Training Loop Start
        for i, (img1_batch, img2_batch, label) in enumerate(train_dataloader):
            img1_batch, img2_batch, label = map(lambda x: x.to(device), [img1_batch, img2_batch, label])
            label = label.view(-1)

            img1_sam = []
            img2_sam = []

            for img1, img2 in zip(img1_batch, img2_batch):
                img1, img2 = img1.to(device), img2.to(device)

                img1_emb = sam(img1.unsqueeze(0))
                img2_emb = sam(img2.unsqueeze(0))

                img1_sam.append(img1_emb.squeeze(0))
                img2_sam.append(img2_emb.squeeze(0))

            img1_batch = torch.stack(img1_sam)
            img2_batch = torch.stack(img2_sam)

            optimizer.zero_grad()

            if contra_loss:
                output1, output2 = model(img1_batch, img2_batch)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                correct_pred += torch.count_nonzero(label == (prob.squeeze(1) > 0.5)).sum().item()
                total_pred += label.size(0)
            else:
                prob = model(img1_batch, img2_batch)
                loss = criterion(prob.squeeze(1), label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                correct_pred += torch.count_nonzero(label == (prob.squeeze(1) > 0.5)).sum().item()
                total_pred += label.size(0)


        train_loss /= len(train_dataloader)
        epoch_train_acc = correct_pred / total_pred

        train_losses.append(train_loss)
        epoch_train_loss = sum(train_losses)/len(train_losses)
        epoch_train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        print("Training: Loss={:.2f} | Accuracy={:.2f}".format(epoch_train_loss, epoch_train_acc))
        # Training Loop End

        # Validation Loop Start
        model.eval()

        correct_pred = 0
        total_pred = 0

        for i, (img1_batch, img2_batch, label) in enumeraet(val_dataloader):
            img1_batch, img2_batch, label = map(lambda x: x.to(device), [img1_batch, img2_batch, label])
            label = label.view(-1)

            img1_sam = []
            img2_sam = []

            for img1, img2 in zip(img1_batch, img2_batch):
                img1, img2 = img1.to(device), img2.to(device)

                img1_emb = sam(img1.unsqueeze(0))
                img2_emb = sam(img2.unsqueeze(0))

                img1_sam.append(img1_emb.squeeze(0))
                img2_sam.append(img2_emb.squeeze(0))

            img1_batch = torch.stack(img1_sam)
            img2_batch = torch.stack(img2_sam)

            if contra_loss:
                with torch.no_grad():
                    output1, output2 = model(img1_batch, img2_batch)
                    loss = criterion(output1, output2, label)

                val_loss += loss.item()
                correct_pred += torch.count_nonzero(label == (prob.squeeze(1) > 0.5)).sum().item()
                total_pred += label.size(0)
            else:
                with torch.no_grad():
                    prob = model(img1_batch, img2_batch)
                    loss = criterion(prob.squeeze(1), label)

                val_loss += loss.item()
                correct_pred += torch.count_nonzero(label == (prob.squeeze(1) > 0.5)).sum().item()
                total_pred += label.size(0)
            
        val_loss /= len(val_dataloader)
        epoch_val_acc = correct_pred / total_pred

        scheduler.step(val_loss)

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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(artifact_path, "best.pt")
            )

        # Plot Loss and Accuracy 
        plot_loss(epoch_train_losses, epoch_val_losses, artifact_path)
        plot_accuracy(train_accs, val_accs, artifact_path)

if __name__ == "__main__":
    train_samsiamese()