import os
import time
from datetime import datetime
from pytz import timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt

def ddp_setup(rank, world_size):
    """
    Function from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int
        ) -> None:
            self.gpu_id = gpu_id
            self.model = model.to(gpu_id)
            self.train_data = train_data
            self.optimizer = optimizer
            self.save_every = save_every
            self.model = DDP(model, device_ids=[gpu_id])
            self.criterion = nn.BCELoss()

            self.epoch_train_losses = []
            self.epoch_val_losses = []
            self.train_losses = []
            self.train_accs = []
            self.val_losses = []
            self.val_accs = []

            self.best_val_loss = 100000000

    def _train_batch(self, source1, source2, targets):
        self.optimizer.zero_grad()
        output = self.model(source1, source2)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _val_batch(self, source1, source2, targets):
        with torch.no_grad():
            output = self.model(source1, source2)
            loss = self.criterion(output, targets)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)


        self.model.train()
        for (source1, source2), targets in self.train_data:
            source1, source2, targets = map(lambda x: x.to(self.gpu_id), [source1, source2, targets])
            self._train_batch(source1, source2, targets)

        self.model.eval()
        for (source1, source2), targets in self.val_data:
            source1, source2, targets = map(lambda x: x.to(self.gpu_id), [source1, source2, targets])
            self._val_batch(source1, source2, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    # train_set =   # load your dataset
    # val_set = 
    # model =   # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)