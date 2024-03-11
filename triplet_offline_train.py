import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch_metric_learning import distances, losses, miners

# Custom Imports
import config
from tensorsiamese import SiameseNetwork
from losses import ContrastiveLoss
from online_pair import SiameseTensorPairDataset
from offline_pair import OfflinePairDataset
from utils import cur_time, write_csv, init_log, init_run_log, create_if_not_exist, load_losses_accs
from eval_metrics import evaluate, plot_loss, plot_accuracy, plot_roc

# Global variables
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
artifact_path = os.path.join(config.sam_bce_out, cur_time())

def main():
    os.makedirs(artifact_path)
    create_if_not_exist(artifact_path)
    init_log(f"{artifact_path}/train.csv")
    init_log(f"{artifact_path}/val.csv")
    init_run_log(f"{artifact_path}/hyperparameters.csv")

    train_pairs = config.combined_train
    val_pairs = config.combined_val
    tensors = config.combined_tensors

    # HYPERPARAMETERS
    # Linear Scaling of learning rate based on [https://arxiv.org/pdf/1706.02677.pdf]
    num_epochs = 30
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
        x:torch.utils.DataLoader(coin_dataset[x],batch_size=batch_size, shuffle=True if x=='train' else False)
        for x in ['train','val']}
    
    model = SiameseNetwork()
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=optim_momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.TripletMarginWithDistanceLoss()

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

        for anchor, positive, negative in dataloaders[phase]:
            anchor, positive, negative = map(lambda x: x.to(device), [anchor, positive, negative])

            # Calculate gradients when phase == 'train'
            with torch.set_grad_enabled(phase == 'train'):
                anc_emb, pos_emb, neg_emb = model(anchor, positive, negative)
                
                # Euclidean distance by default
                loss = F.triplet_margin_with_distance_loss(anc_emb, pos_emb, neg_emb
                                                           )
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                ap_dist = F.pairwise_distance(anc_emb, pos_emb, p=2)
                an_dist = F.pairwise_distance(anc_emb, neg_emb, p=2)

                distances.append(ap_dist.cpu().detach().numpy())
                labels.append(np.ones(ap_dist.size(0)))

                distances.append(an_dist.cpu().detach().numpy())
                labels.append(np.zeros(an_dist.size(0)))

                loss_sum += loss.item()

        avg_loss = loss_sum / len(dataloaders[phase])
        labels = np.array(labels).flatten()
        distances = np.array(distances).flatten()

        tpr, fpr, acc = evaluate(distances, labels)

        accuracy = np.mean(acc)

        print("{}: Loss = {:.8f} | Accuracy = {:.8f}".format(phase, avg_loss, accuracy))

        lr = '_'.join(map(str, scheduler.get_lr()))
        write_csv(f"{artifact_path}/{phase}.csv", [epoch, avg_loss, accuracy, batch_size, lr])

        if phase == 'val':
            return avg_loss, acc
        
if __name__ == "__main__":
    main()