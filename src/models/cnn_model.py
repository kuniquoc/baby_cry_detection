import torch.nn as nn
import torchvision.models as models

class MobileNetV2_Crying(nn.Module):
    def __init__(self):
        super(MobileNetV2_Crying, self).__init__()

        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Chuyển đầu vào thành 1 kênh (MFCC)
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,      # MFCC có 1 kênh

            # Giữ nguyên các thông số khác để đảm bảo tính tương thích với mô hình gốc
            out_channels=32,    # Số lượng kênh đầu ra giữ nguyên theo mô hình gốc để đầu ra phù hợp với đầu vào của lớp tiếp theo
            kernel_size=3,      # kích thước kernel giữ nguyên
            stride=2,           # nhảy 2 pixel để giảm lượng tính toán
            padding=1,          # padding để tranh mất thông tin biên
            bias=False          # Không sử dụng bias vì đã có BatchNorm ở lớp tiếp theo trong mô hình gốc
        )

        # Thay đổi đầu ra: 1000 lớp → 1 lớp nhị phân
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2), # tránh overfitting
            nn.Linear(self.model.last_channel, 1)  # Binary classification
        )

    def forward(self, x):
        return self.model(x).squeeze()