import torch.nn as nn
from cframe.dataloader import *
from cframe.utils import *
from cframe.models import ModelManager
from .basic_inferencer import BasicInferencer
import math


class MunetAttentionInferencer(BasicInferencer):
    def __init__(self,
                 model_manager: ModelManager,
                 dl_manager: BasicDataloaderManager,
                 metric,
                 label='saliency'):
        super(MunetAttentionInferencer, self).__init__(model_manager, dl_manager, metric, label)


    def test(self, dl=None):
        if dl is None:
            dl = self.dl_manager.get_test_dl()

        # loss_meter = AverageMeter()
        # metric_meters = [AverageMeter() for i in range(5)]
        stages = [[] for i in range(5)]
        targets = []

        items = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dl):
                img = data['img']
                # label = data[self.label]
                fixation = data['fixation']
                outs, s = self.model(img)

                # loss = 0
                targets.append(fixation[0].data.cpu().numpy())
                item = []
                for j, out in enumerate(outs):
                    out = torch.squeeze(out, dim=1)
                    stages[j].append(out[0].data.cpu().numpy())
                    cur_acc = self._cal_jud_acc(out[0].cpu(), fixation[0].cpu())
                    item.append(cur_acc)

                # item.append(torch.argmax(s, dim=1) + 1)
                item.extend(s[0].data.cpu().numpy().tolist())
                items.append(item)

            df = pd.DataFrame(items)
            df.to_csv(os.path.join(self.save_dir, 'score.csv'), index=False)
            print(self.save_dir)
        return torch.tensor(stages), torch.tensor(targets)
