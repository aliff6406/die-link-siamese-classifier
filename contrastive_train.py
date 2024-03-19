import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Custom Imports
import config
from samsiamese import SiameseNetwork
from losses import ContrastiveLoss
from datasets import OfflinePairDataset
from utils import cur_time, write_csv, init_log, init_run_log, create_if_not_exist, load_losses_accs
from eval_metrics import evaluate, plot_loss, plot_accuracy, plot_roc

# Global variables
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
artifact_path = os.path.join(config.sam_contrastive_out, cur_time())

def main():
    os.makedirs(artifact_path)
    init_log(f"{artifact_path}/train.csv")
    init_log(f"{artifact_path}/val.csv")
    init_run_log(f"{artifact_path}/hyperparameters.csv")

    train_pairs = config.pair_combined_train
    val_pairs = config.pair_combined_test
    tensors = config.tensor_data

    # HYPERPARAMETERS
    # Linear Scaling of learning rate based on [https://arxiv.org/pdf/1706.02677.pdf]
    num_epochs = 50
    batch_size = 32
    initial_lr = 1e-3
    # weight_decay = 1e-3
    weight_decay = "x"
    optim_momentum = 0.9
    # optim_momentum = "x"
    scheduler_gamma = 0.1
    scheduler_step_size = 5
    

    coin_dataset = {
        'train': OfflinePairDataset(pair_dir=train_pairs, tensor_dir=tensors),
        'val': OfflinePairDataset(pair_dir=val_pairs, tensor_dir=tensors)
    }

    dataloaders = {
        x:torch.utils.data.DataLoader(coin_dataset[x],batch_size=batch_size, shuffle=True if x=='train' else False)
        for x in ['train','val']}
    
    model = SiameseNetwork(contrastive_loss=True)
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=optim_momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = ContrastiveLoss()

    write_csv(f"{artifact_path}/hyperparameters.csv", [num_epochs, initial_lr, batch_size, weight_decay,
                                                    optimizer.__class__.__name__, optim_momentum, 
                                                    scheduler.__class__.__name__, criterion.__class__.__name__, 
                                                    scheduler_step_size, scheduler_gamma])
    
    best_val_loss = 100000000

    for epoch in range(num_epochs):
        print("Epoch [{}/{}]".format(epoch+1, num_epochs))

        start = time.time()
        val_loss, val_acc = train_val(model, optimizer, criterion, epoch, dataloaders, scheduler, batch_size)

        print("Execution Time = {:.2f}".format(time.time() - start))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc
                },
                os.path.join(artifact_path, 'best.pt')
            )

    train_losses, train_accs = load_losses_accs(f"{artifact_path}/train.csv")
    val_losses, val_accs = load_losses_accs(f"{artifact_path}/val.csv")

    plot_loss(train_losses, val_losses, artifact_path)
    plot_accuracy(train_accs, val_accs, artifact_path)

def train_val(model, optimizer, criterion, epoch, dataloaders, scheduler, batch_size):
    for phase in ['train', 'val']:
        
        distances, labels = [], []
        loss_sum = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for tensor1, tensor2, label in dataloaders[phase]:
            tensor1, tensor2, label = map(lambda x: x.to(device), [tensor1, tensor2, label])
            label = label.view(-1)

            # Calculate gradients when phase == 'train'
            with torch.set_grad_enabled(phase == 'train'):
                emb1, emb2 = model(tensor1, tensor2)
                
                loss = criterion(emb1, emb2, label)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                distance = F.pairwise_distance(emb1, emb2, p=2)

                distances.extend(distance.cpu().detach().numpy().flatten())
                labels.extend(label.cpu().detach().numpy().flatten())

                loss_sum += loss.item()

        avg_loss = loss_sum / len(dataloaders[phase])
        labels = np.array(labels).flatten()
        distances = np.array(distances).flatten()

        tpr, fpr, acc, threshold = evaluate(distances, labels)

        accuracy = np.mean(acc)

        print("{}: Loss = {:.4f} | Accuracy = {:.4f}".format(phase, avg_loss, accuracy))

        lr = '_'.join(map(str, scheduler.get_last_lr()))
        write_csv(f"{artifact_path}/{phase}.csv", [epoch, avg_loss, accuracy, batch_size, lr])

        if phase == 'val':
            return avg_loss, acc
        
if __name__ == "__main__":
    main()