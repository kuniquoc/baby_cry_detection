import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DilatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DilatedDepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, 
            groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(residual)
        out = F.relu(out)
        return out


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = DilatedDepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, stride=stride, 
            padding=dilation, dilation=dilation)
        self.conv2 = DilatedDepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=1, 
            padding=dilation, dilation=dilation)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.skip(residual)
        out = F.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Ensure reduction doesn't make channels too small
        reduction_factor = min(reduction, channels // 4)
        reduction_factor = max(reduction_factor, 1)  # At least divide by 1
        
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction_factor, 4), bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(max(channels // reduction_factor, 4), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """Efficient Channel Attention block"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block with expansion factor"""
    def __init__(self, in_channels, out_channels, expansion_factor=4, stride=1, dilation=1):
        super(MBConvBlock, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expansion_factor)
        
        # Expansion phase
        self.expand = nn.Identity()
        if expansion_factor != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False)
            )
            
        # Depthwise phase
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=dilation, 
                    dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=False)
        )
        
        # SE block
        self.se = SEBlock(hidden_dim, reduction=4)
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # Save input for residual
        residual = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze and excitation
        x = self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
            
        return x


class DownsampleBlock(nn.Module):
    """Efficient downsampling with attention"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownsampleBlock, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = ECABlock(out_channels)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.attention(x)
        return x


class LightCryCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(LightCryCNN, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)
        )
        
        # Main feature extraction blocks with increased depth
        # Block 1
        self.block1a = ImprovedResidualBlock(16, 24, stride=2, dilation=1)
        self.se1 = SEBlock(24)
        self.block1b = MBConvBlock(24, 24, expansion_factor=2)
        
        # Block 2
        self.block2a = ImprovedResidualBlock(24, 32, stride=2, dilation=1)
        self.se2 = SEBlock(32)
        self.block2b = MBConvBlock(32, 32, expansion_factor=2)
        
        # Block 3
        self.block3a = ImprovedResidualBlock(32, 48, stride=2, dilation=1)
        self.se3 = SEBlock(48)
        self.block3b = MBConvBlock(48, 48, expansion_factor=2)
        self.block3c = MBConvBlock(48, 48, expansion_factor=2)
        
        # Block 4
        self.block4a = ImprovedResidualBlock(48, 64, stride=2, dilation=2)
        self.se4 = SEBlock(64)
        self.block4b = MBConvBlock(64, 64, expansion_factor=2, dilation=2)
        
        # Block 5
        self.block5a = ImprovedResidualBlock(64, 96, stride=1, dilation=4)
        self.se5 = SEBlock(96)
        self.block5b = MBConvBlock(96, 96, expansion_factor=2, dilation=4)
        
        # Parallel processing paths for multi-scale feature extraction
        self.path1 = nn.Conv2d(96, 32, kernel_size=1, bias=False)  # 1x1 path
        self.path2 = nn.Sequential(  # 3x3 path
            nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        self.path3 = nn.Sequential(  # 5x5 path with dilated conv
            DilatedDepthwiseSeparableConv(96, 32, kernel_size=5, padding=4, dilation=2)
        )
        
        # Final feature compression and attention
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        
        self.final_attention = nn.Sequential(
            ECABlock(128)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate/2),
            nn.Linear(32, 1)  # Binary output
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate and print model parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # Block 1
        x = self.block1a(x)
        x = self.se1(x)
        x = self.block1b(x)
        
        # Block 2
        x = self.block2a(x)
        x = self.se2(x)
        x = self.block2b(x)
        
        # Block 3
        x = self.block3a(x)
        x = self.se3(x)
        x = self.block3b(x)
        x = self.block3c(x)
        
        # Block 4
        x = self.block4a(x)
        x = self.se4(x)
        x = self.block4b(x)
        
        # Block 5
        x = self.block5a(x)
        x = self.se5(x)
        x = self.block5b(x)
        
        # Multi-path feature extraction
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        
        # Concatenate paths
        multi_scale = torch.cat([path1, path2, path3], dim=1)
        
        # Feature fusion and attention
        x = self.feature_fusion(multi_scale)
        x = self.final_attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x.squeeze()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Thêm tùy chọn phát hiện bất thường để debug nếu cần
def enable_anomaly_detection():
    torch.autograd.set_detect_anomaly(True)
    print("Anomaly detection enabled")


# Kiểm tra số lượng tham số
if __name__ == "__main__":
    # Có thể gọi enable_anomaly_detection() để bật chế độ debug
    enable_anomaly_detection()  # Enabling anomaly detection to catch errors
    
    model = LightCryCNN()
    
    # Kiểm tra với sample input
    batch_size = 8
    # Giả sử MFCC có kích thước 44x298
    sample_input = torch.randn(batch_size, 1, 44, 298)
    
    print("Running model forward pass...")
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Ước tính kích thước mô hình (MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)
    total_size = param_size + buffer_size
    
    print(f"Model size: {total_size:.2f} MB")
    print("Model completed successfully!")