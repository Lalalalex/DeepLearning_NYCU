import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channel, cross_channel, out_channel, cross_stride = 1):
        super().__init__()
        self.in_channel = in_channel
        self.cross_channel = cross_channel
        self.out_channel = out_channel
        self.cross_stride = cross_stride
        self.ReLU = nn.ReLU(inplace = True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, cross_channel, kernel_size = (1, 1), stride = (1, 1), bias = False),
            nn.BatchNorm2d(cross_channel, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Conv2d(cross_channel, cross_channel, kernel_size = (3, 3), stride = (cross_stride, cross_stride), padding = (1, 1), bias = False),
            nn.BatchNorm2d(cross_channel, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Conv2d(cross_channel, out_channel, kernel_size = (1, 1), stride = (1, 1), bias = False),
            nn.BatchNorm2d(out_channel, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(inplace = True)
        )
        if in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1), stride = (cross_stride, cross_stride), bias = False),
                nn.BatchNorm2d(out_channel, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
            )
    def forward(self, input):
        if self.in_channel != self.out_channel:
            return self.ReLU(self.layer(input) + self.downsample(input))
        return self.ReLU(self.layer(input) + input)
class ResNet50(nn.Module):
    def __init__(self, out_features = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
        self.bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.ReLU = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode = False)
        self.layer_1 = nn.Sequential(
            Bottleneck(64, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256)
        )
        self.layer_2 = nn.Sequential(
            Bottleneck(256, 128, 512, 2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512)
        )
        self.layer_3 = nn.Sequential(
            Bottleneck(512, 256, 1024, 2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024)
        )
        self.layer_4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features = out_features, bias = True)
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.ReLU(output)
        output = self.maxpool(output)
        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output