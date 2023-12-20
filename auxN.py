import torch.nn as nn
import torch.nn.functional as F

class TransformationPredictor(nn.Module):
    def __init__(self, scale_factor):
        super(TransformationPredictor, self).__init__()
        self.scale_factor = scale_factor

        # conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((85, 85))

        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.adaptive_pool(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.output_conv(x)
        return x
