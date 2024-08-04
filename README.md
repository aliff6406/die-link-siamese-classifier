# Ancient Coin Die Link Classifier

This project utilises Pytorch to train Siamese neural networks for ancient coin classification. The Siamese network utilise Meta AI's Segment Anything Model's image encoder as the backbone to compare its performance to other pretrained models such as AlexNet, ResNet50, ViT_b_16, DenseNet and EfficientNet.

This project uses Segment Anything Model's base vision transformer image encoder which can be downloaded from the link below.
- vit_b: [ViT_B SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## System Requirement & Tech Stack

| Requirements | Description  |
| ------------ | ------------ |
| Python       | Version > 3.8 |

## Development Setup

1. CD into the project folder and run the non-excutable local_setup.sh script to set up the virtual environment

   ```s
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r local_requirements.txt
   ```

### Note on Training/Testing your own Custom Dataset
1.

### Training Settings
1. 
