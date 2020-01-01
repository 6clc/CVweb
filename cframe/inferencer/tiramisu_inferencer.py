import pandas as pd
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from cframe.dataloader import *
from cframe.learner.tools import CyclicLR
from cframe.utils import *
from cframe.utils.fastprogress import *
import os
import shutil
import pickle as pkl
from cframe.models import ModelManager
from .basic_inferencer import BasicInferencer


class TiramisuInferencer(BasicInferencer):
    def __init__(self,
                 model_manager: ModelManager,
                 dl_manager: BasicDataloaderManager,
                 metric,
                 label):
        super(TiramisuInferencer, self).__init__(model_manager, dl_manager, metric, label)

    def test(self, dl=None):
        if dl is None:
            dl = self.dl_manager.get_test_dl()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        items = []

        if self.save_dir is not None:
            assert os.path.exists(self.save_dir)
            gt_dir = os.path.join(self.save_dir, 'gt')
            img_dir = os.path.join(self.save_dir, 'image')
            features_dir = os.path.join(self.save_dir, 'feature')
            pred_dir = os.path.join(self.save_dir, 'pred')
            check_dir(gt_dir)
            check_dir(img_dir)
            check_dir(features_dir)
            check_dir(pred_dir)

        for i, data in tqdm(enumerate(dl)):
            img = data['img']
            label = data[self.label]

            out, feature = self.model(img)

            pred = out.data.cpu().numpy()[0]
            pred = np.transpose(pred, [1, 2, 0])
            pred = np.argmax(pred, axis=-1)

            # feature = [item.data.cpu().numpy()[0] for item in feature]

            loss = self.criteration(out, label.to(self.device))
            loss_meter.update(loss.data.cpu().item())
            cur_metric = self.metric(out.data.cpu().numpy(), label.data.cpu().numpy())
            metric_meter.update(cur_metric)

            name = data['name'][0]
            gt_path = data[self.label + '_path'][0]
            img_path = data['img_path'][0]
            items.append([name, cur_metric])
            if self.save_dir is not None:
                self.save_ans(img_path, img_dir, gt_path, gt_dir, name, pred_dir, pred)

        items.append(['mean_meatric', metric_meter.avg])
        if self.save_dir is not None:
            df = pd.DataFrame(items, columns=['name', 'metric'])
            df.to_csv(os.path.join(self.save_dir, 'score.csv'))

    def save_ans(self, img_path, img_dir, gt_path, gt_dir, name, pred_dir, pred, features_dir=None, feature=None):
        shutil.copy(img_path, img_dir + '/' + (img_path.split('/')[-1]))
        shutil.copy(gt_path, gt_dir + '/' + (gt_path.split('/')[-1]))
        # with open(os.path.join(features_dir, name + '.pkl'), 'wb') as f:
        #     pkl.dump(feature, f)
        pred = pred.astype(np.uint8)
        pred = Image.fromarray(pred)
        pred = pred.convert('P')
        pred.putpalette([0, 0, 0, 255, 255, 255])
        pred.save(os.path.join(pred_dir, name + '.png'))
