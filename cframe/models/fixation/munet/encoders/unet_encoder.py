import torch

from cframe.models.fixation.base import *


class UnetEncoder(nn.Module):
    def __init__(self, nblocks, channels, in_planes):
        super(UnetEncoder, self).__init__()
        self.nblocks = nblocks
        self.channels = channels
        self.in_conv = InConv(in_planes, self.channels[0])

        self.downs = nn.ModuleList()
        for i in range(self.nblocks-1):
            self.downs.append(
                DownSample(self.channels[i], self.channels[i+1])
            )

    def forward(self, x):
        outs = [self.in_conv(x)]
        for i in range(self.nblocks-1):
            outs.append(self.downs[i](outs[-1]))
        return outs


if __name__ == '__main__':
    net = UnetEncoder(nblocks=5, in_planes=3, channels=[64, 128, 256, 512, 512])
    x = torch.rand(size=(2, 3, 224, 224))
    outs = net(x)
    for out in outs:
        print(out.shape)

    # torch.Size([2, 64, 224, 224])
    # torch.Size([2, 128, 112, 112])
    # torch.Size([2, 256, 56, 56])
    # torch.Size([2, 512, 28, 28])
    # torch.Size([2, 512, 14, 14])
