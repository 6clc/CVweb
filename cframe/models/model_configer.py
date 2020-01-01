import torch.nn as nn
from cframe.models.clasification import *
from cframe.models.fixation import *
from cframe.models.fixation.munet.multi_net.multi_unet import MultiUnet
from cframe.models.fixation.munet.multi_net.multi_unet_attention import MultiUnetAttention
from cframe.models.fixation.munet.multi_net.multi_unet_attention_test import MultiUnetAttentionTest
from cframe.models.loss import *
from cframe.models.segment import *

MODEL_DICT = dict(
    fcn16s=dict(name='fcn16s',
                model=FCN16s,
                config=dict(pretrained_backbone=False,
                            n_class=1)),
    munet=dict(name='munet',
               model=MultiUnet,
               num_output=5,
               config=dict(nblocks=5, in_planes=3, num_classes=1,
                           channels=[64, 128, 256, 512, 512])
               ),
    munet_attention=dict(name='munet_attention',
                         model=MultiUnetAttention,
                         config=dict(nblocks=5, in_planes=3, num_classes=1,
                                     channels=[64, 128, 256, 512, 512])
                         ),
    unet_attention=dict(name='unet_attention',
                        model=unet_CT_multi_att_dsv_2D,
                        config=dict(feature_scale=4, n_classes=1, is_deconv=True, in_channels=3,
                                    nonlocal_mode='concatenation', attention_dsample=(2, 2, 2), is_batchnorm=True)),
    munet_attention_test=dict(name='munet_attention_test',
                              model=MultiUnetAttentionTest,
                              config=dict(nblocks=5, in_planes=3, num_classes=1,
                                          channels=[64, 128, 256, 512, 512])
                              ),
    deeplabv3_resnet50=dict(name='deeplabv3_resnet50',
                            model=deeplab_resnet50,
                            config=dict(backbone_name='resnet50',
                                        pretrained_backbone=True,
                                        num_classes=1,
                                        backbone_out_shape=(30, 40),
                                        upsample_factor=16,
                                        gt_shape=(480, 640))),
    tiramisu103=dict(name='tiramisu103',
                     model=FCDenseNet103,
                     config=dict(n_classes=2, is_train=True)),
    tiramisu57=dict(name='tiramisu57',
                    model=FCDenseNet57,
                    config=dict(n_classes=2, is_train=True)
                    ),
    unet_resnet=dict(name='unet_resnet',
                     model=unet_resnet34_cbam_v0a,
                     config=dict(num_classes=4)
                     ),
    resnet_lstm2=dict(name='resnet_lstm2',
                      model=resnet_lstm,
                      config=dict(backbone_name='resnet50',
                                  pretrained_backbone=True)),
    se_densenet161=dict(name='se_densenet161',
                        model=se_densenet161,
                        config=dict(pretrained=True, num_classes=2)),
    se_densenet201=dict(name='se_densenet201',
                        model=se_densenet201,
                        config=dict(pretrained=True, num_classes=4)),
    resnet_lstm_vertical=dict(name='resnet_lstm_vertical',
                              model=resnet_lstm_vertical,
                              sub_dir='hidden_512-num_3',
                              config=dict(backbone_name='resnet50',
                                          pretrained_backbone=True,
                                          in_channels=512,
                                          hidden_channels=[512, 512, 512],
                                          num_classes=1,
                                          gt_shape=(480, 640)
                                          )),
    resnet_lstm_horizontal=dict(name='resnet_lstm_horizontal',
                                model=resnet_lstm_horizontal,
                                config=dict(backbone_name='resnet50',
                                            pretrained_backbone=True,
                                            in_channels=[512, 1024, 2048],
                                            hidden_channels=[64, 64, 64],
                                            num_repeat=3,
                                            num_classes=1)),
    resnet_lstm_horizontal_vertical=dict(name='resnet_lstm_horizontal',
                                         model=resnet_lstm_horizontal,
                                         config=dict(backbone_name='resnet50',
                                                     pretrained_backbone=True,
                                                     in_channels=[512, 1024, 2048],
                                                     hidden_channels=[64, 64, 64],
                                                     num_repeat=3,
                                                     num_classes=1)),
)

LOSS_DICT = dict(
    bce=dict(name='bce',
             loss=nn.BCELoss,
             config=dict(weight=None, size_average=None, reduce=None, reduction='mean')),
    ce=dict(name='ce',
            loss=CrossEntropyLoss2d,
            config=dict(weight=None, size_average=None, ignore_index=-100,
                        reduce=None, reduction='mean'
                        )),
    mse=dict(name='mse',
             loss=nn.MSELoss,
             config=dict(size_average=None, reduce=None, reduction='mean')),
    kl=dict(name='kl',
            loss=nn.KLDivLoss,
            config=dict(size_average=None, reduce=None, reduction='mean')),
    kl_sam=dict(name='kl_sam',
                loss=KlDivergence,
                config=dict(config=None)),
    cc_sam=dict(name='cc_sam',
                loss=CorrelationCoefficient,
                config=dict(config=None)),
    nss_sam=dict(name='nss_sam',
                 loss=NSS,
                 config=dict(config=None)),
    sam_loss=dict(name='sam_loss',
                  loss=SamLoss,
                  config=dict(config=None)),
    SymmetricLovaszLoss=dict(name='SymmetricLovaszLoss',
                             loss=SymmetricLovaszLoss,
                             config=(dict(weight=None, size_average=True)))
)

model_name = MODEL_DICT.keys()
loss_name = LOSS_DICT.keys()


class ModelConfiger(object):
    @classmethod
    def get_model_names(cls):
        return model_name

    @classmethod
    def get_loss_names(cls):
        return loss_name

    @classmethod
    def get_model_config(cls, name):
        return MODEL_DICT[name]

    @classmethod
    def get_loss_config(cls, name):
        return LOSS_DICT[name]
