import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_planes, out_planes)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_planes, out_planes, bilinear=True):
        super(UpSample, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_planes // 2, in_planes // 2, 2, stride=2)

        self.conv = DoubleConv(in_planes, out_planes)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY//2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_planes, out_planes)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, 1)
        # self.score = nn.Softmax(dim=2)
        # self.score = nn.Sigmoid()
        # self.score = nn.ReLU()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1),
            # nn.ReLU(inplace=True)
            # nn.BatchNorm2d(out_planes),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        # N, C, H, W = x.size()
        # output_score = self.score(x.view(N, C, -1))
        # output_score = output_score.contiguous().view(N, C, H, W)
        # x = self.sigmoid(x)
        # out = self.score(out)
        return out






