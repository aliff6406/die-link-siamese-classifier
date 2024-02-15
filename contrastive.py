import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on the paper: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        # Experiment with margin value
        self.margin = margin

    def forward(self, emb1, emb2, label):
        euclidean_dist = F.pairwise_distance(emb1, emb2, keepdim=True)
        print("Euclidean Distance: ", euclidean_dist)
        positive = (label) * torch.pow(euclidean_dist, 2)
        print("Positive Loss: ", positive)
        negative = (1-label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2)
        print("negative Loss: ", negative)
        loss_contrastive = torch.mean(positive + negative)
        return loss_contrastive
