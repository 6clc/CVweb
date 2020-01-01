import torch.nn as nn
import torch
from cframe.models.base_module.conv_lstm import ConvLSTM
from torch.nn import functional as F
from cframe.models.utils import vgg_backbone


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
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
        x = self.conv(x)
        out, _ = self.convlstm(x)
        return out[-1]


class VggLstm(nn.Module):
    def __init__(self, backbone, n_classes=1):
        super().__init__()
        vgg13_backbone_out = [64, 128, 256, 512, 512]
        vgg13_backbone_out = vgg13_backbone_out[::-1]
        self.backbone = backbone

        self.bottle_lstm = ConvLSTM(
            in_channels=vgg13_backbone_out[0],
            hidden_channels=vgg13_backbone_out[0],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.bottle_neck = UpBlock(vgg13_backbone_out[0], vgg13_backbone_out[0])

        self.up_blocks = nn.ModuleList()
        for i in range(1, 4):
            self.up_blocks.append(
                ConvLSTM(
                    in_channels=2*vgg13_backbone_out[i],
                    hidden_channels=2*vgg13_backbone_out[i],
                    kernel_size=(3, 3),
                    num_layers=1,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False
                )
            )
            self.up_blocks.append(
                UpBlock(in_channels=2 * vgg13_backbone_out[i],
                        out_channels=vgg13_backbone_out[i + 1])
            )
        self.final = ConvLSTM(
            in_channels=vgg13_backbone_out[-1]*2,
            hidden_channels=n_classes,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

    def forward(self, x):
        outs = self.backbone(x)
        # print(type(outs), outs.keys())
        # outs = outs.values()
        x = self.bottle_lstm(outs[4])
        x = self.bottle_neck(x)

        for i in range(3):
            x = self.up_blocks[i](torch.cat([x, outs[3-i]]))
        x = self.final(x)

        return x


if __name__ == '__main__':
    configer = dict(
        backbone_name='vgg13',
        pretrained_backbone=False
    )
    img = torch.rand(size=(2, 3, 224, 224)).cuda()
    # model = resnet_lstm(configer).backbone.cuda()
    # model = resnet_lstm(configer).cuda()
    # model = deeplabv3_resnet50(pretrained=False).backbone
    # print(model)
    backbone = vgg_backbone(configer)
    model = VggLstm(backbone=backbone).cuda()
    outs = model(img)
    # for k, v in out.items():
    #     print(k, v.shape)
    # for k, v in outs.items():
    #     print(k, v.shape)
    # for out in outs:
    #     print(out.shape)
    print(outs.shape)

# resnet50 lstm with dilation
# 0 torch.Size([2, 512, 28, 28])
# 1 torch.Size([2, 1024, 28, 28])
# 2 torch.Size([2, 2048, 28, 28])

# deeplabv3 resnet50 backbone
# out torch.Size([2, 2048, 28, 28])

# vgg 13
# 0 torch.Size([2, 64, 224, 224])
# 1 torch.Size([2, 128, 112, 112])
# 2 torch.Size([2, 256, 56, 56])
# 3 torch.Size([2, 512, 28, 28])
# 4 torch.Size([2, 512, 14, 14])
