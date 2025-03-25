import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = None  # Placeholder, set dynamically

        self.fc1 = None  # Will be initialized in `forward()`
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flatten_dim(self, x):
        """Helper function to calculate the flattened dimension dynamically"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return torch.flatten(x, start_dim=1).shape[1]


    def forward(self, x):
        if self.flatten_dim is None: 
            with torch.no_grad():
                self.flatten_dim = self._get_flatten_dim(torch.zeros(1, x.shape[1], x.shape[2], x.shape[3]).to(x.device))
                self.fc1 = nn.Linear(self.flatten_dim, 256).to(x.device)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits
