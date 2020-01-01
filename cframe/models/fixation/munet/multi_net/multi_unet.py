import torch.nn as nn

from ..decoders import *
from ..encoders import *


class MultiUnet(nn.Module):
    def __init__(self, configer):
        super(MultiUnet, self).__init__()
        self.configer = configer
        self.nblocks = self.configer['nblocks']
        self.channels = self.configer['channels']
        self.in_planes = self.configer['in_planes']
        self.num_classes = self.configer['num_classes']

        self.encoder = UnetEncoder(self.nblocks, channels=self.channels, in_planes=self.in_planes)
        self.decoders = nn.ModuleList()

        for i in range(self.nblocks):
            self.decoders.append(
                UnetDecoder(i+1, self.num_classes, self.channels[:i+1])
            )

    def forward(self, x):
        encoder_outs = self.encoder(x)
        decoder_outs = []
        for i in range(self.nblocks):
            decoder_outs.append(self.decoders[i](encoder_outs[:i+1]))
        return decoder_outs


if __name__ == '__main__':
    print(MultiUnet)
    net = MultiUnet().cuda()
    print(net)
    x = torch.randn((2, 3, 32, 32)).cuda()
    outs = net(x)
    for out in outs:
        print(out.shape)
