import torch

from cframe.models.fixation.base import *


class UnetDecoder(nn.Module):
    def __init__(self, nblocks, num_classes, channels):
        super(UnetDecoder, self).__init__()
        self.nblocks = nblocks
        self.channels = channels
        self.channels.reverse()

        self.ups = nn.ModuleList()
        if self.nblocks == 2:
            self.ups.append(UpSample(self.channels[0] + self.channels[1], self.channels[1]))
        if self.nblocks > 2:
            self.ups.append(UpSample(self.channels[0] + self.channels[1], self.channels[2]))
            for i in range(2, self.nblocks-1):
                self.ups.append(
                    UpSample(self.channels[i]*2, self.channels[i+1])
                )
            self.ups.append(UpSample(self.channels[-1]*2, self.channels[-1]))
        self.out_conv = OutConv(self.channels[-1], num_classes)

    def forward(self, ins):
        if self.nblocks == 1:
            out = ins[0]
        else:
            ins.reverse()
            out = self.ups[0](ins[0], ins[1])
            for i in range(1, self.nblocks-1):
                out = self.ups[i](out, ins[i+1])
        return self.out_conv(out)


if __name__ == '__main__':
    outs = []
    outs.append(torch.rand(2, 64, 224, 224))
    outs.append(torch.rand(2, 128, 112, 112))
    outs.append(torch.rand(2, 256, 56, 56))
    outs.append(torch.rand(2, 512, 28, 28))
    outs.append(torch.rand(2, 512, 14, 14))

    channels = [64, 128, 256, 512, 512]

    scale = 2
    net = UnetDecoder(scale, 1, channels=channels[:scale])
    out = net(outs[:scale])
    print(out.shape)