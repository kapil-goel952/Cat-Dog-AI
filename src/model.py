
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CatDogCNN(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.backbone = resnet18(weights=weights)

        # Freeze the pretrained layers initially.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Allow the last feature-extraction block to learn.
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True

        # Replace the original classifier with our two classes.
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)
