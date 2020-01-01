import torch.nn as nn


class ModelManager(object):
    def __init__(self, model_config, loss_config, device_ids=None):
        self.model_config = model_config
        self.loss_config = loss_config
        self.device_ids = device_ids

        self.model = None
        self.loss = None
        self._init()

    def _init(self):
        self.model = self.model_config['model'](self.model_config['config'])
        self.loss = self.loss_config['loss'](**self.loss_config['config'])
        if self.device_ids is None:
            self.model = self.model.to('cpu')
            self.loss = self.loss.to('cpu')
        else:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
            self.loss = self.loss.cuda()

    def get_model(self):
        return self.model

    def get_loss(self):
        return self.loss

