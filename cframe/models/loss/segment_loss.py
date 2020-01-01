import torch.nn as nn
from .base_loss import lovasz_losses as L
from torch.nn import functional as F
class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    def forward(self, logits, targets):
        return ((L.lovasz_hinge(logits, targets, per_image=True)) \
                + (L.lovasz_hinge(-logits, 1-targets, per_image=True))) / 2


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                        reduce=None, reduction='mean'):
        weight=None
        size_average=True
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, data_dict):
        logits = data_dict['pred']
        if 'segment' in data_dict:
            targets = data_dict['segment']
        elif 'label' in data_dict:
            targets = data_dict['label']
        return self.nll_loss(F.log_softmax(logits), targets)
