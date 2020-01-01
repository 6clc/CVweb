import torch.nn as nn
from cframe.dataloader import *
from cframe.utils import *
import os
from cframe.models import ModelManager
from cframe.metrics.metric_manager import get_metric_funcs

class BasicInferencer(object):
    def __init__(self, learner, after_func, metrics=None):
        self.model_manager = learner.model_manager
        self.model = learner.model
        self.criteration = learner.criteration
        self.dl_manager = learner.dl_manager
        self.after_func = after_func
        self.learner = learner

        if metrics is None:
            print('the function of inferencer is', learner.metrics)
            self.metrics = learner.metrics
            self.metric_funcs = learner.metric_funcs
        else:
            self.metrics = metrics
            self.metric_funcs, _ = get_metric_funcs(metrics)
        self.save_dir = None
        self.summary_writer_dir = learner.summary_writer_dir
        self.task = learner.task

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self._init()

    def _init(self):
        # root_dir = '/'.join(self.dl_manager.configer['root_dir'].split('/')[:-1]) + '/ScoreWriter'
        # data_name = self.dl_manager.config['data_name']
        # model_name = self.model_manager.model_config['name']
        # loss_name = self.model_manager.loss_config['name']
        # self.save_dir = os.path.join(root_dir, '_'.join([data_name, self.task, model_name, loss_name]))
        self.save_dir = self.summary_writer_dir.replace('SummaryWriter', 'ScoreWriter')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def test(self, dl=None):
        raise NotImplementedError

    def load(self, name, path=None):
        if path is None:
            path = self.summary_writer_dir
        path = os.path.join(path, name+'.pth')

        para = torch.load(path, map_location='cpu')
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(para['model'])
        else:
            self.model.load_state_dict(para['model'])