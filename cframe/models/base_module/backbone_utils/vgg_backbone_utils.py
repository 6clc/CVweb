from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import vgg
import torch


def vgg13_backbone(configer):
    pretrained_backbone = configer['pretrained_backbone']
    backbone = vgg.__dict__['vgg13'](
        pretrained=pretrained_backbone
    ).features
    return_layers = {'3': 0, '8': 1, '13': 2, '18': 3, '23': 4}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return backbone


def vgg16_backbone(configer):
    pretrained_backbone = configer['pretrained_backbone']
    backbone = vgg.__dict__['vgg16'](
        pretrained=pretrained_backbone
    ).features
    return_layers = {'4': 0, '9': 1, '16': 2, '23': 3, '30': 4}
    # print(backbone)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return backbone


def vgg_backbone(configer):
    backbone_name = configer['backbone_name']
    if backbone_name == 'vgg13':
        return vgg13_backbone(configer)
    elif backbone_name == 'vgg16':
        return vgg16_backbone(configer)
    else:
        raise KeyError('backbone name {} dosent exist'.format(backbone_name))


if __name__ == '__main__':
    configer = dict(
        pretrained_backbone=False,
        backbone_name='vgg16'
    )
    model = vgg_backbone(configer).cuda()
    x = torch.randn(4, 3, 224, 224).cuda()
    outs = model(x)
    for k, v in outs.items():
        print(k, v.shape)
# vgg 16
# 0 torch.Size([4, 64, 112, 112])
# 1 torch.Size([4, 128, 56, 56])
# 2 torch.Size([4, 256, 28, 28])
# 3 torch.Size([4, 512, 14, 14])
# 4 torch.Size([4, 512, 7, 7])