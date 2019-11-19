import torch
import torch.nn as nn
import numpy as np

class AlexNet(nn.Module):
    def __init__(self, image_shape, num_classes):
        super(AlexNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # Compute visual encoder output dim
        features_size = self.get_output_dim(self.feature_extractor, image_shape)
        self.classifier = nn.Sequential(
            nn.Linear(features_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        return prediction

    def get_output_dim(self, model, image_dim):
        return model(torch.rand(1, *(image_dim))).data.view(1, -1).size(1)