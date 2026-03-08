import torch
import torch.nn as nn

class HVFCnnDigitizer(nn.Module):
    """
    A lightweight 4-layer Convolutional Neural Network for digitizing 
    Humphrey Visual Field reports, as described in:
    "High-Accuracy Digitization of Humphrey Visual Field Reports 
    Using Convolutional Neural Networks" by Shie & Su (2025).
    """
    def __init__(self, num_classes=72):
        super(HVFCnnDigitizer, self).__init__()
        
        # Input size is 48x48 pixel grayscale image (1 channel)
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 24 x 24
            
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 12 x 12
            
            # Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 6 x 6
            
            # Layer 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 3 x 3
        )
        
        # Classifier producing exactly 72 output categories
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): grayscale image patches of shape (B, 1, 48, 48)
        Returns:
            torch.Tensor: logits of shape (B, 72)
        """
        x = self.features(x)
        logits = self.classifier(x)
        return logits
