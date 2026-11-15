
import torch
import torch.nn as nn


class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third convolution block
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model = CatDogCNN()

    # Test using one sample image
    sample = torch.randn(1, 3, 128, 128)

    output = model(sample)

    print("Model created successfully!")
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
