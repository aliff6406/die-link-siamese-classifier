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
        # Modified Contrastive Loss function where similar pairs have a label of 1 and dissmilar pairs have a label of 0
        # Euclidean / L2 norm
        dist = F.pairwise_distance(emb1, emb2, p=2)

        # Manhattan / L1 norm
        # dist = F.pairwise_distance(emb1, emb2, p=1)

        loss = (label) * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)
        return loss