# import torch.nn as nn
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# class Aspp(nn.Sequential):
#     super().__init__()
#     self.head = DeepLabHead(in_channels=self.hidden_channels[-1], num_classes=self.num_classes)
#     self.bn_rlu = nn.Sequential(
#         nn.BatchNorm2d(num_classes),
#         nn.ReLU(inplace=True)
#     )