import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential (
            # nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm2d(64),
            # nn.MaxPool2d(2, 2),

            # nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm2d(128),
            # nn.MaxPool2d(2, 2),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm2d(256),
            # nn.MaxPool2d(2, 2),

            # nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm2d(512),

            # nn.Flatten(),
            # nn.Linear(131072, 1024),
            # nn.ReLU(inplace=True),
            # # nn.BatchNorm1d(1024)
            nn.Flatten(),
        )

        self.fc = nn.Sequential (
            # nn.Linear(1024, 1),
            nn.Linear(49152, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.abs(output1 - output2)
        output = self.fc(output)    
        return output
