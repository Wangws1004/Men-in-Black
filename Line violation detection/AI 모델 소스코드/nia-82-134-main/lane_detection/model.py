from torch import nn
from torchvision.models.segmentation import fcn_resnet50


class LaneSegModel(nn.Module):
    def __init__(self, num_classes=21):
        super(LaneSegModel, self).__init__()
        self.fcn = fcn_resnet50(pretrained=True)
        in_channels = 2048
        inter_channels = in_channels // 4
        channels = num_classes
        self.num_lanes = num_classes
        self.fcn.classifier = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),  # yellow, white, blue, shoulder
        )
        self.f1 = 0
        self.f1cnt = 0

    def forward(self, x):
        out = self.fcn(x)
        return out
