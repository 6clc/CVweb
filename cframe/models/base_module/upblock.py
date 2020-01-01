import torch.nn as nn
import torch.nn.functional as F
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return self.relu(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = ConvBn2d(in_channels, out_channels)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.upsample(x, scale_factor=self.scale_factor)
        return self.conv(x)