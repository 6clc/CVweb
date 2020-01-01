import torch.nn as nn
from cframe.models.base_module.backbone_utils import *
import torch
from cframe.models.base_module.conv_lstm import ConvLSTM
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F

class Conv2d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3)):
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, padding_mode='same'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class CnnLstmVertical(nn.Module):
    def __init__(self, backbone, in_channels,
                 hidden_channels=None, num_classes=1, gt_shape=(224, 224), backbone_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 64, 128]
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(self.hidden_channels)
        self.num_classes = num_classes


        self.conv_feats = nn.ModuleList()
        for i in range(3):
            self.conv_feats.append(
                Conv2d(backbone_channels[i], self.in_channels)
            )

        self.convlstm = ConvLSTM(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=(3, 3),
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.head = DeepLabHead(in_channels=self.hidden_channels[-1], num_classes=self.num_classes)
        self.bn_rlu = nn.Sequential(
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        self.gt_shape = gt_shape
        self.out_num = 3

    def free_backbone_layers(self, layers):
        if 0 in layers:
            print('freeze resnet init layers')
            for para in self.backbone.conv1.parameters():
                para.requires_grad = False
            for para in self.backbone.bn1.parameters():
                para.requires_grad = False
        if 1 in layers:
            print('freeze resnet layer1')
            for para in self.backbone.layer1.parameters():
                para.requires_grad = False
        if 2 in layers:
            print('freeze resnet layer2')
            for para in self.backbone.layer2.parameters():
                para.requires_grad = False
        if 3 in layers:
            print('freeze resnet layer3')
            for para in self.backbone.layer3.parameters():
                para.requires_grad = False
        if 4 in layers:
            print('freeze resnet layer4')
            for para in self.backbone.layer4.parameters():
                para.requires_grad = False

    def set_out_num(self, num):
        self.out_num = num

    def forward(self, x):

        outs = self.backbone(x)
        outs = list(outs.values())
        N, C, H, W = outs[-1].shape
        outs = outs[:self.out_num]

        for i in range(len(outs)):
            # old version: cat all vertical featrue maps
            # n, c, h, w = outs[i].shape
            # outs[i] = torch.cat([outs[i], torch.zeros((n, C-c, h, w)).to(outs[i].device)], dim=1)
            outs[i] = self.conv_feats[i](outs[i])

        out = torch.stack(outs, dim=1)
        out, _ = self.convlstm(out)
        out = out[-1]

        x = self.head(out[:, -1, :, :, :])
        x = F.interpolate(x, size=self.gt_shape, mode='bilinear', align_corners=False)
        return self.bn_rlu(x)


def resnet_lstm_vertical(config):
    backbone = deeplab_resnet(config)
    in_channels = config['in_channels']
    hidden_channels = config['hidden_channels']
    num_classes = config['num_classes']
    gt_shape = config['gt_shape']
    backbone_channels = [512, 1024, 2048]
    model = CnnLstmVertical(backbone,
                            in_channels,
                            hidden_channels,
                            num_classes, gt_shape=gt_shape, backbone_channels=backbone_channels)
    return model


def vgg_lstm_vertical(config):
    raise NotImplementedError
    # backbone = segm_resnet(config)
    # in_channels = config['in_channels']
    # hidden_channels = config['hidden_channels']
    # num_classes = config['num_classes']
    # model = CnnLstmVertical(backbone, in_channels, hidden_channels, num_classes)
    # return model

