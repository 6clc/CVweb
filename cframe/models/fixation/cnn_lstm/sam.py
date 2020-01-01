import torch.nn as nn
from cframe.models.utils.backbone_utils import segm_resnet
import torch
from cframe.models.base_module.conv_lstm import ConvLSTM
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F
from cframe.models.utils import vgg_backbone


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

class Sam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.convlstm = ConvLSTM(
            in_channels=2048,
            hidden_channels=64,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.head = DeepLabHead(in_channels=64, num_classes=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        outs = self.backbone(x)
        outs = list(outs.values())
        out = outs[-1]
        out = torch.stack([out, out, out, out], dim=1)
        # print(out.shape)
        out, _ = self.convlstm(out)
        out = out[-1]
        x = self.head(out[:, -1, :, :, :])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def sam(config):
    backbone = segm_resnet(config)
    model = Sam(backbone)
    return model


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
    model = vgg_backbone(configer).cuda()
    outs = model(img)
    # for k, v in out.items():
    #     print(k, v.shape)
    for k, v in outs.items():
        print(k, v.shape)
    # for out in outs:
    #     print(out.shape)

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
