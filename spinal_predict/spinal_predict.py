root_dir = '/home/hanshan'
data_root_dir = root_dir + '/Data/DataCV'
summary_root_dir = root_dir + '/SummaryWriter'


from cframe.dataloader import SegmentDataloaderManager
from cframe.dataloader import DataConfiger

from cframe.models import ModelManager
from cframe.models import ModelConfiger
from cframe.learner import Learner

from cframe.utils.summary_writer import SummaryWriter
from cframe.metrics.metric_manager import get_metirc_name

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import os
import warnings
import pandas as pd
import cv2
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
has_path=os.path.exists

DataConfiger.set_data_root_dir(data_root_dir)
# DataConfiger.get_all_data_name()

data_config = DataConfiger.get_data_config('LumbarSpinal')
data_config['label_name'] = 'segment'
data_config['resize'] = (224, 224)
data_config['train']['batch_size'] = 4
data_config['valid']['batch_size'] = 4
# data_config
dl_manager = SegmentDataloaderManager(data_config)

model_config = ModelConfiger.get_model_config('unet_resnet')
model_config['config']['n_classes'] = 4
# model_config['config']['gt_shape'] = (480, 640)
# model_config

loss_config = ModelConfiger.get_loss_config('ce')
# loss_config
model_manager = ModelManager(model_config, loss_config, device_ids=[0])

optimizer = Adam(model_manager.model.parameters(), lr=3e-3,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

scheduler = StepLR(optimizer, step_size=5, gamma=0.33)

learner = Learner(model_manager=model_manager,
                          dl_manager=dl_manager,
                          metrics=['iou'],
                          scheduler=scheduler,
                          optimizer=optimizer, label=['segment'], task='Segment')

learner.load('best')

from cframe.inferencer import Inferencer
from cframe.inferencer import multi_metrics
from cframe.inferencer import save_fixations

inferencer = Inferencer(learner, metrics=['iou'],
                        after_func=save_fixations)

spinal_predict = inferencer.predict

