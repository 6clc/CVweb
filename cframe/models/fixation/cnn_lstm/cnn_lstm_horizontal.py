import torch.nn as nn
from cframe.models.base_module.backbone_utils import *
import torch
from cframe.models.base_module.conv_lstm import ConvLSTM
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F


class CnnLstmHorizontal(nn.Module):
    def __init__(self, backbone, in_channels, hidden_channels=None, num_repeat=3, num_classes=1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 64, 64]
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.num_out_layers = len(self.in_channels)
        self.num_lstm_layers = len(self.hidden_channels)
        self.num_classes = num_classes
        self.num_repeat = num_repeat

        self.convlstms = nn.ModuleList()
        for i in range(self.num_out_layers):
            if i == 0:
                cur_inchannel = self.in_channels[i] * self.num_repeat
            else:
                cur_inchannel = (self.in_channels[i] + self.hidden_channels[-1]) * self.num_repeat
            self.convlstms.append(ConvLSTM(in_channels=cur_inchannel,
                                           hidden_channels=self.hidden_channels,
                                           kernel_size=(3, 3),
                                           num_layers=self.num_lstm_layers,
                                           batch_first=True,
                                           return_all_layers=False))

        self.head = DeepLabHead(in_channels=self.hidden_channels[-1], num_classes=self.num_classes)
        self.bn_rlu = nn.Sequential(
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        outs = self.backbone(x)
        outs = list(outs.values())

        for i in range(len(outs)):
            if i == 0:
                x = outs[i]
            else:
                x = torch.cat([outs[i], out], dim=1)

            x = torch.stack([x for j in range(self.num_repeat)], dim=1)
            out, _ = self.convlstms[i](x)
            out = out[-1][:, -1, :, :, :]

        x = self.head(out)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return self.bn_rlu(x)


def resnet_lstm_horizontal(config):
    backbone = deeplab_resnet(config)
    in_channels = config['in_channels']
    hidden_channels = config['hidden_channels']
    num_classes = config['num_classes']
    model = CnnLstmHorizontal(backbone, in_channels, hidden_channels, num_classes)
    return model


def vgg_lstm_vertical(config):
    raise NotImplementedError
    # backbone = segm_resnet(config)
    # in_channels = config['in_channels']
    # hidden_channels = config['hidden_channels']
    # num_classes = config['num_classes']
    # model = CnnLstmVertical(backbone, in_channels, hidden_channels, num_classes)
    # return model
