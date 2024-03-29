import os
import time
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginWithDistanceLoss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Custom Imports
import config
from tripletsiamese import SiameseNetwork
from triplet_tensorpair import SiameseTripletTensorDataset

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

def train(train_dataloader, model, criterion, optimizer, device, train_losses, epoch_train_losses, epoch_train_accs, total_train_loss, train_samples):
    model.train()
    correct_pred = 0
    total_pred = 0
    train_distances = []
    train_labels = []

    for anchor_tensor, positive_tensor, negative_tensor in train_dataloader:
        anchor_tensor, positive_tensor, negative_tensor = map(lambda x: x.to(device), [anchor_tensor, positive_tensor, negative_tensor])

        anchor, positive, negative = model(anchor_tensor, positive_tensor, negative_tensor)
        loss = criterion(anchor, positive, negative)

        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        correct_pred += torch.count_nonzero(pos_distance < neg_distance).sum().item()
        total_pred += anchor_tensor.size(0)
        train_samples += anchor_tensor.size(0)

    total_train_loss /= len(train_dataloader)
    epoch_train_acc = correct_pred / total_pred

    train_losses.append(total_train_loss)
    epoch_train_loss = sum(train_losses)/len(train_losses)
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_accs.append(epoch_train_acc)

    print("Training: Loss={:.2f} | Accuracy={:.2f} | Train Triples={}".format(epoch_train_loss, epoch_train_acc, train_samples))
    
def validate(val_dataloader, model, criterion, device, scheduler, val_losses, epoch_val_accs, epoch_val_losses, total_val_loss, val_samples):
    model.eval()    
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for anchor_tensor, positive_tensor, negative_tensor in val_dataloader:
            anchor_tensor, positive_tensor, negative_tensor = map(lambda x: x.to(device), [anchor_tensor, positive_tensor, negative_tensor])

            anchor, positive, negative = model(anchor_tensor, positive_tensor, negative_tensor)
            loss = criterion(anchor, positive, negative)

            pos_distance = F.pairwise_distance(anchor, positive)
            neg_distance = F.pairwise_distance(anchor, negative)

            total_val_loss += loss.item()

            correct_pred += torch.count_nonzero(pos_distance < neg_distance).sum().item()
            total_pred += anchor_tensor.size(0)
            val_samples += anchor_tensor.size(0)

    total_val_loss /= len(val_dataloader)
    epoch_val_acc = correct_pred / total_pred

    val_losses.append(total_val_loss)
    epoch_val_accs.append(epoch_val_acc)

    epoch_val_loss = sum(val_losses)/len(val_losses)
    epoch_val_losses.append(epoch_val_loss)

    print("Validation: Loss={:.2f} | Accuracy={:.2f} | Val Triplets={}".format(epoch_val_loss, epoch_val_acc, val_samples))

    return epoch_val_loss


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

    # HYPERPARAMETERS
    # Linear Scaling of learning rate based on [https://arxiv.org/pdf/1706.02677.pdf]
    batch_size = 32
    num_epochs = 100
    # base_lr = 5e-2
    # learning_rate = base_lr * batch_size/256
    learning_rate = 1e-3
    # weight_decay = 1e-6

    train_dataset = SiameseTripletTensorDataset(label_dir=train_csv, tensor_dir=train_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = SiameseTripletTensorDataset(label_dir=val_csv, tensor_dir=val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build Model
    model = SiameseNetwork()
    model.to(device)

    # Optimizer, LR Scheduler and Criterion (Loss Function)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = TripletMarginWithDistanceLoss()

    best_val_loss = 100000000

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    training_times = []

    for epoch in range(num_epochs):
        print("Epoch [{} / {}]".format(epoch+1, num_epochs))
        epoch_start = time.time()

        total_val_loss = 0.0
        total_train_loss = 0.0
        train_losses = []
        val_losses = []
        train_samples = 0
        val_samples = 0

        train(train_dataloader, model, criterion, optimizer, device, train_losses, epoch_train_losses, epoch_train_accs, total_train_loss, train_samples)
        val_loss = validate(val_dataloader, model, criterion, device, scheduler, val_losses, epoch_val_accs, epoch_val_losses, total_val_loss, val_samples)

        scheduler.step()
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        training_times.append(epoch_time)
        print("Epoch Training Time: {:.2f}".format(epoch_time))


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
        plot_accuracy(epoch_train_accs, epoch_val_accs, artifact_path)
    
    print("Average Epoch Training Time: {:.2f} | Total Training Time: {:.2f}".format(sum(training_times) / len(training_times), sum(training_times)))

if __name__ == "__main__":
    main()