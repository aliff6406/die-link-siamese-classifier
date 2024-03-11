import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch_metric_learning import distances, losses, miners, reducers

# Custom Imports
import config
from tensorsiamese import SiameseNetwork
from losses import ContrastiveLoss
from datasets import OnlineTripletDataset, OfflineTripletDataset
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

    combined_labels = config.combined_train
    val_triplets = config.combined_val
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
    triplet_type = "semihard" # can be from ["all", "semihard", "hard"]


    train_dataset = OnlineTripletDataset(label_dir=combined_labels, tensor_dir=tensors),
    val_dataset = OfflineTripletDataset(triplet_dir=val_triplets, tensor_dir=tensors)

    train_dataloader = torch.utils.Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.Dataloader(val_dataset, batch_size=batch_size, shuffle=True)

    ### pytorch-metric-learning ###
    reducer = reducers.ThresholdReducer(low=0)
    criterion = losses.TripletMarginLoss(margin=1.0, reducer=reducer)
    triplet_miner= miners.TripletMarginMiner(margin=1.0, type_of_triplets=triplet_type)
    ### pytorch-metric-learning ###
    
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
        train(model, optimizer, criterion, epoch, train_dataloader, triplet_miner, scheduler, batch_size)
        val_loss, val_acc = val(model, epoch, val_dataloader, scheduler, batch_size)

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

def train(model, optimizer, criterion, epoch, dataloader, triplet_miner, scheduler, batch_size):
    distances, labels_issame = [], []
    num_mined_triplets = 0
    loss_sum = 0.0

    model.train()
    for batch_idx, (tensor, label) in enumerate(dataloader):
        tensor, label = map(lambda x: x.to(device), [tensor, label])

        embedding = model.forward_once(tensor)
        indices_tuple = triplet_miner(embedding, label)
        loss = criterion(embedding, label, indices_tuple)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, triplet_miner.num_triplets))

        loss_sum += loss
        num_mined_triplets += triplet_miner.num_triplets
    
    avg_loss = loss_sum / num_mined_triplets

    print("train: loss = {:.8f} | accuracy = {:.8f}".format(avg_loss, 0))

    lr = '_'.join(map(str, scheduler.get_lr()))
    write_csv(f"{artifact_path}/train.csv", [epoch, avg_loss, 0, batch_size, lr])


def val(model, epoch, dataloader, scheduler, batch_size):
    distances, labels = [], []
    loss_sum = 0.0

    criterion = nn.TripletMarginWithDistanceLoss()
    model.eval()
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anc_emb, pos_emb, neg_emb = model(anchor, positive, negative)

            loss = criterion(anc_emb, pos_emb, neg_emb)
        
            ap_dist = F.pairwise_distance(anc_emb, pos_emb, p=2)
            an_dist = F.pairwise_distance(anc_emb, neg_emb, p=2)

            distances.append(ap_dist.cpu().detach().numpy())
            labels.append(np.ones(ap_dist.size(0)))

            distances.append(an_dist.cpu().detach().numpy())
            labels.append(np.zeros(an_dist.size(0)))

            loss_sum += loss.item()

        avg_loss = loss_sum / len(dataloader)
        labels = np.array(labels).flatten()
        distances = np.array(distances).flatten()

        tpr, fpr, acc = evaluate(distances, labels)

        accuracy = np.mean(acc)

        print("val: Loss = {:.8f} | Accuracy = {:.8f}".format(avg_loss, accuracy))

        lr = '_'.join(map(str, scheduler.get_lr()))
        write_csv(f"{artifact_path}/val.csv", [epoch, avg_loss, accuracy, batch_size, lr])

        return avg_loss, accuracy
    
if __name__ == "__main__":
    main()