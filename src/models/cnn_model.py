import torch.nn as nn
import torchvision.models as models

class MobileNetV2_Crying(nn.Module):
    def __init__(self):
        super(MobileNetV2_Crying, self).__init__()

        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)

        # ✅ Chuyển đầu vào thành 1 kênh (MFCC)
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,      # MFCC có 1 kênh
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # ✅ Thay đổi đầu ra: 1000 lớp → 1 lớp nhị phân
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2), #nên dùng dropout để tránh overfitting, thử nghiệm với 0.5
            nn.Linear(self.model.last_channel, 1)  # Binary classification
        )

    def forward(self, x):
        return self.model(x).squeeze()