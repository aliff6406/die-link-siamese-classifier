import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="vit_b_16"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models 
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)


        # Get the number of features that are outputted by the last layer of backbone network.

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward_once(self, img):
        feats = self.vit._process_input(img)

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = self.vit.encoder(feats)

        # We're only interested in the representation of the classifier token that we appended at position 0
        feats = feats[:, 0]
        print(feats.shape)
        return feats

    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.forward_once(img1)
        feat2 = self.forward_once(img2)

        print("feat1 shape: ", feat1.shape)
        print("feat2 shape: ", feat2.shape)
        
        # Multiply (element-wise) the feature vectors of the two images together, 
        # to generate a combined feature vector representing the similarity between the two.
        abs_diff = (feat1 - feat2)

        # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
        output = self.cls_head(abs_diff)
        return output
    
# testing
if __name__ == "__main__":
    SiameseNetwork()