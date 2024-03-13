import os
import time
import argparse

import torch
import torch.nn as nn

# Custom Imports
import config
from siamese import SiameseNetwork
from datasets import OfflineImagePairDataset
from utils import cur_time, write_csv, init_log, init_run_log, create_if_not_exist, load_losses_accs
from eval_metrics import evaluate_bce, evaluate, plot_loss, plot_accuracy, plot_roc

# Global variables
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
artifact_path = os.path.join(config.vgg_bce_out, cur_time())

def main():
    os.makedirs(artifact_path)
    create_if_not_exist(artifact_path)
    init_log(f"{artifact_path}/train.csv")
    init_log(f"{artifact_path}/val.csv")
    init_run_log(f"{artifact_path}/hyperparameters.csv")

    train_pairs = config.img_pair_combined_train
    val_pairs = config.img_pair_combined_val
    images = config.img_data

    # HYPERPARAMETERS
    # Linear Scaling of learning rate based on [https://arxiv.org/pdf/1706.02677.pdf]
    num_epochs = 50
    batch_size = 32
    initial_lr = 1e-3
    # weight_decay = 1e-5
    weight_decay = "x"
    optim_momentum = 0.9
    # optim_momentum = "x"
    scheduler_gamma = 0.1
    scheduler_step_size = 5
    

    coin_dataset = {
        'train': OfflineImagePairDataset(pair_dir=train_pairs, img_dir=images, transform='resnet50'),
        'val': OfflineImagePairDataset(pair_dir=val_pairs, img_dir=images, transform='resnet50')
    }

    dataloaders = {
        x:torch.utils.data.DataLoader(coin_dataset[x],batch_size=batch_size, shuffle=True if x=='train' else False)
        for x in ['train','val']}
    
    model = SiameseNetwork(backbone='vgg16')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=optim_momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    criterion = nn.BCELoss()

    write_csv(f"{artifact_path}/hyperparameters.csv", [num_epochs, initial_lr, batch_size, weight_decay,
                                                    optimizer.__class__.__name__, optim_momentum, 
                                                    scheduler.__class__.__name__, criterion.__class__.__name__, 
                                                    scheduler_step_size, scheduler_gamma])
    
    best_val_loss = 100000000

    for epoch in range(num_epochs):
        print("Epoch [{}/{}]".format(epoch+1, num_epochs))

        start = time.time()
        val_loss, val_acc = train_val(model, optimizer, criterion, epoch, dataloaders, scheduler, batch_size)

        # scheduler.step()

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
        
        preds, labels = [], []
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
                prob = model(tensor1, tensor2)
                loss = criterion(prob.squeeze(1), label)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                loss_sum += loss.item()
                preds.extend((prob > 0.5).cpu().detach().numpy().flatten())
                labels.extend(label.cpu().detach().numpy().flatten())

        avg_loss = loss_sum / len(dataloaders[phase])
        tpr, fpr, acc = evaluate_bce(preds, labels)

        print("{}: Loss = {:.4f} | Accuracy = {:.4f}".format(phase, avg_loss, acc))

        lr = '_'.join(map(str, scheduler.get_last_lr()))
        write_csv(f"{artifact_path}/{phase}.csv", [epoch, avg_loss, acc, batch_size, lr])
        
        if phase == 'val':
            return avg_loss, acc
        
if __name__ == "__main__":
    main()