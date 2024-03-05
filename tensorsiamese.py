import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self,
                 contrastive_loss=None,
                 cosine=None,
                 ):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.cosine = cosine

        # Convolutional Layers for input of size ([b, 256, 16, 16]) 
        self.conv_layers = nn.Sequential(
            # Input 256 x 16 x 16
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),

            # nn.Flatten(start_dim=1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1),
        )

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if inputted similarity
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*8*8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward_once(self, input):
        output = self.conv_layers(input)
        return output
    
    def forward(self, tensor1, tensor2):
        output1 = self.forward_once(tensor1)
        output2 = self.forward_once(tensor2)
        if self.contrastive_loss:
            return output1, output2
        else:
            # Test Euclidean Distance / Cosine Similarity / Absolute Difference / Dot Product
            # Iteration 1: Absolute Difference
            abs_diff = torch.abs(output1 - output2)
            output = self.cls_head(abs_diff)
            return output
