import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Layers for input of size ([b, 256, 16, 16]) 
        self.conv_layers = nn.Sequential(
            # Input 256 x 16 x 16
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),

            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),

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

    def forward_once(self, input):
        emb = self.conv_layers(input)
        return emb
    
    def forward(self, anchor_tensor, positive_tensor, negative_tensor):
        anchor = self.conv_layers(anchor_tensor)
        positive = self.conv_layers(positive_tensor)
        negative = self.conv_layers(negative_tensor)

        return anchor, positive, negative