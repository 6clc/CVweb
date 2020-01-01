from cframe.models.fixation.munet.decoders import *
from cframe.models.fixation.munet.encoders import *


class MultiUnetAttention(nn.Module):
    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.nblocks = self.configer['nblocks']
        self.channels = self.configer['channels']
        self.in_planes = self.configer['in_planes']
        self.num_classes = self.configer['num_classes']

        self.encoder = UnetEncoder(self.nblocks, channels=self.channels, in_planes=self.in_planes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention = nn.Linear(sum(self.channels), 64)
        self.attention_fc = nn.Linear(64, 5)
        self.gate = nn.Sigmoid()

        self.decoders = nn.ModuleList()
        for i in range(self.nblocks):
            self.decoders.append(
                UnetDecoder(i+1, self.num_classes, self.channels[:i+1])
            )

    def forward(self, x):
        encoder_outs = self.encoder(x)

        s = []
        for i in range(len(encoder_outs)):
            s.append(self.avg_pool(encoder_outs[i]))
        s = torch.cat(s, dim=1)
        B, C, H, W = s.shape
        s = s.squeeze()
        s = s.reshape((B, C))
        s = self.attention(s)
        s = self.attention_fc(s)
        s = self.gate(s)
        # for i in range(len(encoder_outs)):
        #     encoder_outs[i] = encoder_outs[i]*(s[:, i].view(-1, 1, 1, 1).expand_as(encoder_outs[i]))

        decoder_outs = []
        for i in range(self.nblocks):
            decoder_outs.append(self.decoders[i](encoder_outs[:i+1]))
        return decoder_outs, s


if __name__ == '__main__':
    config = dict(nblocks=5, in_planes=3, num_classes=1,
                  channels=[64, 128, 256, 512, 512])
    net = MultiUnetAttention(config)
    if torch.cuda.is_available():
        net = net.cuda()
    x = torch.randn((2, 3, 32, 32))
    if torch.cuda.is_available():
        x = x.cuda()
    outs = net(x)
    for out in outs:
        print(out.shape)
