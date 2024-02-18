import os

import torch
import torch.nn as nn

from segment_anything import sam_model_registry

class SiameseNetworkSAM(nn.Module):
    def __init__(self,
                 contrastive_loss=None
                 ):
        '''
            Parameters:
                model_type (str, optional): The model type. It can be one of the following: vit_h, vit_l, vit_b.
                    Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
                device (str, optional): The device to use. It can be one of the following: cpu, cuda.
                    Defaults to None, which will use cuda if available.
                checkpoint_dir (str, optional): The path to the model checkpoint. It can be one of the following:
                    sam_vit_h_4b8939.pth, sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth.
                    Defaults to None. See https://bit.ly/3VrpxUh for more details.
        '''
        super().__init__()
        self.contrastive_loss = contrastive_loss

        # number of features of flattened SAM image encoder output
        out_features = 256 * 64 * 64

        self.contra_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, 1024),
            nn.ReLU(inplace=True)
        )

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if inputted similarity
        self.cls_head = nn.Sequential(
            nn.Linear(out_features, 1024),
            # nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),

            # nn.Dropout(p=0.5),
            nn.Linear(512, 64),

            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[1, 3, 1024, 1024]
                    img2 (torch.Tensor): shape=[1, 3, 1024, 1024]

            batch size = 1, SAM does not support batch inputs as of 11/02/2024

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass both images through SAM's image encoder and get the feature vectors
        # The image embeddings returned is a tensor with shape 1xCxHxW where C is
        # the embedding dimension and (H,W) are the embedding spatial dimension of SAM
        # C=256, H=W=64
        if self.contrastive_loss:
            # Run SAM output through contra_head
            output1 = self.contra_head(img1)
            output2 = self.contra_head(img2)
            return output1, output2
        else:
            input1 = torch.flatten(img1)
            input2 = torch.flatten(img2)
            # Test Euclidean Distance / Cosine Similarity / Absolute Difference / Dot Product
            # Iteration 1: Absolute Difference
            abs_diff = torch.abs(input1 - input2)
            # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
            output = self.cls_head(abs_diff)
            return output