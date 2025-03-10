import torch
import torchvision
import torch.nn  as nn
import torch.nn.functional as F
import numpy as np


class Resnet18_model(nn.Module):
    def __init__(self,
                 in_channels : int,
                 num_classes : int
                 ) -> None:
        super(Resnet18_model, self).__init__()
        self.in_channels = in_channels
        self.upsampling_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=3,
            kernel_size=1
        )
        
        self.resnet_layer = torchvision.models.resnet18(
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        )

        self.resnet_layer.fc = nn.Linear(
            in_features=self.resnet_layer.fc.in_features,
            out_features=num_classes
        )

    def forward(self, x) :
        if self.in_channels != 3 : 
            x = F.relu(self.upsampling_layer(x))

        x = self.resnet_layer(x)
        logits = F.softmax(x, dim=1)

        return logits 


