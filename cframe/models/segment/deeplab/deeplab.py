import torch.nn as nn
from cframe.models.base_module.backbone_utils import *
import torch
from torch.nn import functional as F


class DeeplabResnet50(nn.Module):
    def __init__(self, backbone, num_classes, upsample_factor, backbone_out_shape, gt_shape):
        super().__init__()
        in_channels = 2048
        self.num_classes = num_classes
        self.backbone_out_shape = backbone_out_shape
        self.gt_shape = gt_shape
        self.backbone = backbone
        self.upsample_factor = upsample_factor

        self.conv_feat = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding_mode='same'),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(512, num_classes, 1, padding_mode='same'),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # self.upsample = UpBlock(num_classes, num_classes,
#                                scale_factor=self.upsample_factor)

        # self.head = DeepLabHead(in_channels=in_channels, num_classes=self.num_classes)

    def forward(self, x):
        outs = self.backbone(x)
        outs = list(outs.values())
        out = outs[-1]
        # print(out.shape)
        out = self.conv_feat(out)
        # print(out.shape)
        out = self.conv_final(out)
        # print(out.shape)
        out = F.upsample_bilinear(out, size=self.gt_shape)
        return out


def deeplab_resnet50(config):
    backbone = deeplab_resnet(config)
    backbone_out_shape = config['backbone_out_shape']
    gt_shape = config['gt_shape']
    num_classes = config['num_classes']
    upsample_factor = config['upsample_factor']
    model = DeeplabResnet50(backbone, num_classes, upsample_factor, backbone_out_shape, gt_shape)
    return model


def vgg_lstm_vertical(config):
    raise NotImplementedError
    # backbone = segm_resnet(config)
    # in_channels = config['in_channels']
    # hidden_channels = config['hidden_channels']
    # num_classes = config['num_classes']
    # model = CnnLstmVertical(backbone, in_channels, hidden_channels, num_classes)
    # return model


if __name__ == '__main__':
    config = dict(backbone_name='resnet50',
                  pretrained_backbone=True,
                  num_classes=1,
                  backbone_out_shape=(30, 40),
                  gt_shape=(480, 640),
                  upsample_factor=16)
    model = deeplab_resnet50(config).cuda()
    x = torch.randint(255, size=(4, 3, 240, 320), dtype=torch.float).cuda()
    y = model(x)
    # print(model)
    print(y.shape)

