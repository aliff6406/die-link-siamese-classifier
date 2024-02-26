import torch
import torch.nn as nn

from segment_anything import sam_model_registry

class SAMImageEncoderViT(nn.Module):
    def __init__(self, 
                 model_type="vit_b",
                 device=None,
                 checkpoint=None,
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
        self.model_type = model_type
        self.device = device
        self.checkpoint = checkpoint
        self.contrastive_loss = contrastive_loss

        # Use cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device)

        for params in self.sam.parameters():
            params.requires_grad = False

        self.conv_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d((16,16)),
        )

    def forward(self, img):
        with torch.no_grad():
            img = self.sam.preprocess(img)
            output = self.sam.image_encoder(img)
        output = self.conv_layers(output)

        return output