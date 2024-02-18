import os
from datetime import datetime
from pytz import timezone
import config

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from tensor_pair_dataset import SiameseTensorPairDataset
from contrastive import ContrastiveLoss
from tensorsiamese import SiameseNetworkSAM

def cur_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    uk_time = timezone('Europe/London')
    loc_dt = datetime.now(uk_time)
    return loc_dt.strftime(fmt).replace(' ', '_')

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        contra_loss=None
    ) -> None:
        self.gpu_id = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.contra_loss= contra_loss

        if contra_loss:
            self.criterion = ContrastiveLoss(margin=1.0)
        else:
            self.criterion = nn.BCELoss()

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        (tensor1, tensor2), label = data
        tensor1, tensor2, label = tensor1.to(self.gpu_id), tensor2.to(self.gpu_id), label.to(self.gpu_id)
        if self.contra_loss:
            output1, output2 = self.model(tensor1, tensor2)
            loss = self.criterion(output1, output2, label.view(-1))
            loss.backwards()
            self.optimizer.step()
        else:
            output = self.model(tensor1, tensor2)
            loss = self.criterion(output, label.view(-1))
            loss.backward()
            self.optimizer.step()

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for data in self.train_data:
            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        if self.gpu_id == 0:
            ckp = self.model.module.state_dict()
            PATH = f"checkpoint_epoch_{epoch}.pt"
            torch.save(ckp, PATH)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            if self.gpu_id == 0:
                print("Epoch: [{} / {}]".format(epoch+1, max_epochs))
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    # Here you would load your actual SiameseNetworkSAM model and dataset
    train_set = SiameseTensorPairDataset(label_dir=config.obverse_train_csv, tensor_dir=config.obverse_train_dir)
    model = SiameseNetworkSAM(contrastive_loss=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, contra_loss: bool):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, contra_loss=None)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    total_epochs = 100
    save_every = 10
    batch_size = 1
    contra_loss = False

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size, contra_loss), nprocs=world_size)