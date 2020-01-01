import pandas as pd
import torch.nn as nn
from torch import optim
from cframe.dataloader import *
from cframe.learner.tools import CyclicLR
from cframe.utils import *
from cframe.models import ModelManager
from cframe.metrics.metric_manager import get_metric_funcs


class BasicLearner(object):
    def __init__(self,
                 model_manager: ModelManager,
                 dl_manager,
                 metrics, optimizer, scheduler, label, task):

        self.model_manager = model_manager
        self.model = self.model_manager.get_model()
        self.criteration = self.model_manager.get_loss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label = label

        self.dl_manager = dl_manager
        self.train_dl = dl_manager.get_train_dl()
        self.valid_dl = dl_manager.get_valid_dl()
        # self.test_dl = dl_manager.get_test_dl()

        self.summary_writer_dir = None
        self.writer = None
        self.task = task

        self.running_score = dict()
        self.metrics = metrics
        self.metric_names = '-'.join(self.metrics)
        self.metric_funcs, self.batch_metric_funcs = get_metric_funcs(metrics)
        self.best_para = None
        self.num_output = 1
        if 'num_output' in self.model_manager.model_config:
            self.num_output = self.model_manager.model_config['num_output']

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self._init()

    def _init(self):
        root_dir = '/'.join(self.dl_manager.config['root_dir'].split('/')[:-1])+'/SummaryWriter'
        data_name = self.dl_manager.config['data_name']
        model_name = self.model_manager.model_config['name']
        loss_name = self.model_manager.loss_config['name']
        if 'sub_dir' in self.model_manager.model_config:
            sub_dir = self.model_manager.model_config['sub_dir']
            dir_name = '-'.join([model_name, loss_name, sub_dir])
        else:
            dir_name = ']['.join([model_name, loss_name])
        self.summary_writer_dir = os.path.join(root_dir, self.task, data_name, dir_name)
        if not os.path.exists(self.summary_writer_dir):
            os.makedirs(self.summary_writer_dir)
        print('the log of train will be logged in ', self.summary_writer_dir)

    def fit_one_cycle(self, t, base_lr, max_lr, multi=2, log_nth=None, show_batch_loss=False):
        self.optimizer = optim.SGD(self.model.parameters(), lr=base_lr)
        self.scheduler = CyclicLR(self.optimizer, base_lr, max_lr,
                                  step_size_up=len(self.train_dl))
        num_epoches = int(t * multi * 2)
        self.train(num_epoches, log_nth, show_batch_loss)

    def train(self, num_epoches, log_nth=None, show_batch_loss=False):
        raise NotImplementedError

    def _train_validate(self, mb, epoch, num_epoches, log_nth, show_batch_loss):
        raise NotImplementedError

    def predict(self, **data):
        img = data['img']
        img = Image.open(img)
        img = np.array(img)

        img = self.dl_manager.img_transform(img)

        self.model.eval()
        img = torch.unsqueeze(img, dim=0)
        outs = self.model(img)
        self.model.train()
        return outs

    def get_para(self):
        if type(self.model) == nn.DataParallel:
            model_para = self.model.module.state_dict()
        else:
            model_para = self.model.state_dict()

        optim_para = self.optimizer.state_dict()
        para = dict(model=model_para, optim=optim_para)
        return para

    def save(self, name='best', para=None,  path=None):
        if para is None:
            assert self.best_para is not None
            para = self.best_para
        if path is None:
            path = self.summary_writer_dir
            if not name.endswith('.pth'):
                name += '.pth'
            path = os.path.join(path, name)

        torch.save(para, path)

    def load(self, name, path=None, pth_start_module=False):
        if path is None:
            path = self.summary_writer_dir
        path = os.path.join(path, name + '.pth')
        para = torch.load(path, map_location='cpu')

        if 'model' not in para.keys():
            model_para = para
            if type(self.model) == nn.DataParallel and not pth_start_module:
                self.model.module.load_state_dict(model_para)
            elif type(self.model) == nn.DataParallel and pth_start_module:
                self.model.load_state_dict(model_para)
            else:
                self.model.load_state_dict(model_para)
        else:
            model_para = para['model']
            optim_para = para['optim']

            if type(self.model) == nn.DataParallel:
                self.model.module.load_state_dict(model_para)
            else:
                self.model.load_state_dict(model_para)

            self.optimizer.load_state_dict(optim_para)
