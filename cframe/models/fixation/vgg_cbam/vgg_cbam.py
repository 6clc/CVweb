import torch.nn as nn
from cframe.models.utils import vgg_backbone
from cframe.models.attention_cbam import CBAM

class VggCbam166(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config['backbone_name'] = 'vgg16'
        self.backbone = vgg_backbone(config)
        self.n_class = config['n_class']
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        pass