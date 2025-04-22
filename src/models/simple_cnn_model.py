import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block with Conv2d, BatchNorm, ReLU, and MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class SimpleCNN_Crying(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN_Crying, self).__init__()
        
        # Similar to MobileNetV2_Crying, this model takes 1-channel input (MFCC)
        self.features = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32),    # Block 1
            ConvBlock(in_channels=32, out_channels=64),   # Block 2
            ConvBlock(in_channels=64, out_channels=128),  # Block 3
            ConvBlock(in_channels=128, out_channels=256), # Block 4
        )
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier similar to MobileNetV2_Crying
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)  # Binary classification
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze()
    
    def count_parameters(self):
        """Calculate and return the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage:
if __name__ == "__main__":
    model = SimpleCNN_Crying()
    num_params = model.count_parameters()
    print(f"Number of trainable parameters: {num_params:,}")
