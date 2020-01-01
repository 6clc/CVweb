from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet


def maskrcnn_resnet(backbone_name, pretrained, replace_stride_with_dilation):
    raise NotImplementedError


def deeplab_resnet(configer):
    backbone_name = configer['backbone_name']
    pretrained_backbone = configer['pretrained_backbone']
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True]
    )

    return_layers = {'layer2': 0, 'layer3': 1, 'layer4': 2}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return backbone
