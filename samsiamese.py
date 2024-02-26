import torch
import torch.nn as nn

from segment_anything import sam_model_registry

class SiameseNetworkSAM(nn.Module):
    def __init__(self, contrastive_loss=None):
        super().__init__()
        self.contrastive_loss = contrastive_loss

        # Use cuda if available
        # if device is None:
            # device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build SAM model
        # self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        # self.sam.to(device)

        # for params in self.sam.parameters():
        #     params.requires_grad = False

        self.conv_layers = nn.Sequential(
            # nn.AdaptiveMaxPool2d((16,16)),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1),
        )

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if inputted similarity
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32*16*16, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            # nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            # nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, input):
        output = self.conv_layers(input)
        return output

    def forward(self, intput1, intput2):
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        if self.contrastive_loss:
            return output1, output2
        else:
            # Test Euclidean Distance / Cosine Similarity / Absolute Difference / Dot Product
            # Iteration 1: Absolute Difference
            abs_diff = torch.abs(output1 - output2)
            output = self.cls_head(abs_diff)
            return output