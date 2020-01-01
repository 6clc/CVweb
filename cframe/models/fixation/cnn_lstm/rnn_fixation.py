import torch.nn as nn
from cframe.models.base_module.backbone_utils import *
import torch
from cframe.models.base_module.conv_lstm import ConvLSTM
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F



class RNNFixation(nn.Module):
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
        N, C, H, W = outs[-1].shape
        for i in range(len(outs)):
            n, c, h, w = outs[i].shape
            outs[i] = torch.cat([outs[i], torch.zeros((n, C-c, h, w)).to(outs[i].device)], dim=1)
        out = torch.stack(outs, dim=1)
        # print(out.shape)
        out, _ = self.convlstm(out)
        out = out[-1]
        x = self.head(out[:, -1, :, :, :])
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


def resnet_lstm(config):
    backbone = deeplab_resnet(config)
    model = RNNFixation(backbone)
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
