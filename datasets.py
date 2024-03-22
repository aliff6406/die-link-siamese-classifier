import os

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OfflinePairDataset(Dataset):
    def __init__(self, pair_dir, tensor_dir, shuffle_pairs=True):
        self.pair_df = pd.read_csv(pair_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        tensor1_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 0])
        tensor2_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 1])
        label = self.pair_df.iloc[idx, 2]

        tensor1 = torch.load(tensor1_path, map_location='cpu').squeeze()
        tensor2 = torch.load(tensor2_path, map_location='cpu').squeeze()

        return tensor1, tensor2, torch.tensor(label, dtype=torch.float32)
    
class OfflineTripletDataset(Dataset):
    def __init__(self, triplet_dir, tensor_dir, shuffle_pairs=True):
        self.triplet_df = pd.read_csv(triplet_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.triplet_df)
    
    def __getitem__(self, idx):
        anc_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 0])
        pos_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 1])
        neg_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 2])

        anchor = torch.load(anc_path, map_location='cpu').squeeze()
        positive = torch.load(pos_path, map_location='cpu').squeeze()
        negative = torch.load(neg_path, map_location='cpu').squeeze()

        return anchor, positive, negative
    
class OnlineTripletDataset(Dataset):
    '''
    Complete Combined / Obverse / Reverse Dataset to be used with Triplet Miner

    '''
    def __init__(self, label_dir, tensor_dir):
        self.label_df = pd.read_csv(label_dir, header=None)
        self.tensor_dir = tensor_dir

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.label_df.iloc[idx, 0])
        label = self.label_df.iloc[idx, 1]

        tensor = torch.load(tensor_path, map_location='cpu').squeeze()

        return tensor, label

class OfflineImagePairDataset(Dataset):
    def __init__(self, pair_dir, img_dir, transform=None):
        self.pair_df = pd.read_csv(pair_dir, header=None)
        self.img_dir = img_dir

        if transform is not None:
            if transform == "resnet50":
                # Apply normalization for Resnet50
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
                
    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        img1_path = os.path.join(self.img_dir, self.pair_df.iloc[idx, 0])
        img2_path = os.path.join(self.img_dir, self.pair_df.iloc[idx, 1])
        label = self.pair_df.iloc[idx, 2]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1).float()
            img2 = self.transform(img2).float()

        return img1, img2, torch.tensor(label, dtype=torch.float32)

class EvaluatePairDataset(Dataset):
    def __init__(self, pair_dir, tensor_dir, shuffle_pairs=True):
        self.pair_df = pd.read_csv(pair_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        tensor1_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 0])
        tensor2_path = os.path.join(self.tensor_dir, self.pair_df.iloc[idx, 1])
        label = self.pair_df.iloc[idx, 2]

        tensor1 = torch.load(tensor1_path, map_location='cpu').squeeze()
        tensor2 = torch.load(tensor2_path, map_location='cpu').squeeze()

        tensor1_name = os.path.splitext(self.pair_df.iloc[idx, 0])[0]
        tensor2_name = os.path.splitext(self.pair_df.iloc[idx, 1])[0]
        return tensor1, tensor2, torch.tensor(label, dtype=torch.float32), tensor1_name, tensor2_name

class EvaluateImagePairDataset(Dataset):
    def __init__(self, pair_dir, img_dir, transform=None):
        self.pair_df = pd.read_csv(pair_dir, header=None)
        self.img_dir = img_dir

        if transform is not None:
            if transform == "resnet50":
                # Apply normalization for Resnet50
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
                
    def __len__(self):
        return len(self.pair_df)
    
    def __getitem__(self, idx):
        img1_path = os.path.join(self.img_dir, self.pair_df.iloc[idx, 0])
        img2_path = os.path.join(self.img_dir, self.pair_df.iloc[idx, 1])
        label = self.pair_df.iloc[idx, 2]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        coin1 = os.path.splitext(self.pair_df.iloc[idx, 0])[0]
        coin2 = os.path.splitext(self.pair_df.iloc[idx, 1])[0]
        
        if self.transform:
            img1 = self.transform(img1).float()
            img2 = self.transform(img2).float()

        return img1, img2, torch.tensor(label, dtype=torch.float32), coin1, coin2
    
class EvaluateTripletDataset(Dataset):
    def __init__(self, triplet_dir, tensor_dir, shuffle_pairs=True):
        self.triplet_df = pd.read_csv(triplet_dir, header=None)
        self.tensor_dir = tensor_dir
        self.shuffle_pairs = shuffle_pairs

    def __len__(self):
        return len(self.triplet_df)
    
    def __getitem__(self, idx):
        anc_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 0])
        pos_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 1])
        neg_path = os.path.join(self.tensor_dir, self.triplet_df.iloc[idx, 2])

        anchor_coin = os.path.splitext(self.triplet_df.iloc[idx, 0])[0]
        positive_coin = os.path.splitext(self.triplet_df.iloc[idx, 1])[0]
        negative_coin = os.path.splitext(self.triplet_df.iloc[idx, 2])[0]

        anchor = torch.load(anc_path, map_location='cpu').squeeze()
        positive = torch.load(pos_path, map_location='cpu').squeeze()
        negative = torch.load(neg_path, map_location='cpu').squeeze()

        return anchor, positive, negative, anchor_coin, positive_coin, negative_coin