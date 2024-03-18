import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self,
                 contrastive_loss=None,
                 backbone=None,
                 ):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        
        # Default embedding number of features for SAM backbone
        emb_num_features = 128 * 8 * 8

        if backbone is not None:
            self.backbone_name = backbone
            if backbone not in models.__dict__:
                raise Exception("Model named {} does not exist in torchvision.models.".format(self.backbone))
            if backbone == "resnet50":
                print("Training with ResNet50 backbone")
                # Resnet50 pretrained on ImageNet1k v1
                resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
                for param in self.backbone.parameters():
                    param.requires_grad = False
                emb_num_features = 128 * 7 * 7
                feature_map_size = 512

            elif backbone == "vgg16":
                print("Training with VGG16 backbone")
                # VGG16 with batch normalization pretrained on ImageNet1k v1
                # Number of trainable parameters: 7082496
                vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(vgg16.children())[:-2])
                for param in self.backbone.parameters():
                    param.requires_grad = False
                # # Unfreeze layers in Sequential(0) from layer (34) to end
                # for param in self.backbone[0][34].parameters():
                #     param.requires_grad = True
                emb_num_features = 128 * 7 * 7
                feature_map_size = 512

            elif backbone == "alexnet":
                print("Training with AlexNet backbone")
                # AlexNet pretrained on ImageNet1k v1
                # Number of trainable parameters if few, hence follow architecture of using SAM backbone
                alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(alexnet.children())[:-2])
                # Freeze all AlexNet layers
                for param in self.backbone.parameters():
                    param.requires_grad = True
                emb_num_features = 128 * 6 * 6
                feature_map_size = 256
            
            elif backbone == "efficientnet":
                print("Training with EfficientNet_b0 backbone")
                # EfficientNet_b0 pretrained on ImageNet1k v1
                # Model too large to fine-tune, add conv_layers and cls_head similar to SAM backbone
                efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])
                for param in self.backbone.parameters():
                    param.requires_grad = False
                emb_num_features = 128 * 7 * 7
                feature_map_size = 512

            elif backbone == "densenet121":
                print("Training with DenseNet-121 backbone")
                # DenseNet121 pretrained on ImageNet1k v1
                densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(densenet.children())[:-1])
                for param in self.backbone.parameters():
                    param.requires_grad = False
                emb_num_features = 128 * 7 * 7
                feature_map_size = 512

            elif backbone == "squeezenet1_1":
                print("Training with SqueezeNet1_1 backbone")
                # SqueezeNet1_1 pretrained on ImageNet1k v1
                squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

            elif backbone == "mobilenet_v2":
                print("Training with MobileNet_v2 backbone")
                # MobileNet pretrained on ImageNet1k v1
                mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
                for param in self.backbone.parameters():
                    param.requires_grad = False
                emb_num_features = 128 * 7 * 7


        # Reduce number of feature maps using 1x1 convolutions
        self.downsample_resnet50 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, stride=1)
        )

        self.downsample_efficientnetb0 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1, stride=1)
        )

        self.downsample_densenet121 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        )

        # Convolutional layers for other pre-trained models
        self.stacked_conv = nn.Sequential(
            nn.Conv2d(in_channels=feature_map_size, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1)
        )

        # Convolutional Layers for SAM output of size ([b, 256, 16, 16]) 
        self.conv_layers = nn.Sequential(
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

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1)
        )

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Create an MLP (multi-layer perceptron) as the classification head. 
        # Classifies if inputted similarity
    
    def forward_once(self, input):
        if self.backbone is not None:
            emb = self.backbone(input)
            if self.backbone_name == "resnet50":
                out = self.downsample_resnet50(emb)
                output = self.stacked_conv(out)
            elif self.backbone_name == "vgg16":
                output = self.stacked_conv(emb)
            elif self.backbone_name == "alexnet":
                output = self.stacked_conv(emb)
            elif self.backbone_name == "efficientnet":
                out = self.downsample_efficientnetb0(emb)
                output = self.stacked_conv(out)
            elif self.backbone_name == "densenet121":
                out = self.downsample_densenet121(emb)
                output = self.stacked_conv(out)
            elif self.backbone_name == "mobilenet_v2":
                out = self.downsample_efficientnetb0(emb)
                output = self.stacked_conv(out)
        else:
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
