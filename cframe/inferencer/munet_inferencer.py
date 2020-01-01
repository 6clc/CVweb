import torch.nn as nn
from cframe.dataloader import *
from cframe.utils import *
from cframe.models import ModelManager
from .basic_inferencer import BasicInferencer
import math


class MunetInferencer(BasicInferencer):
    def __init__(self,
                 model_manager: ModelManager,
                 dl_manager: BasicDataloaderManager,
                 metric,
                 label='saliency'):
        super(MunetInferencer, self).__init__(model_manager, dl_manager, metric, label, after_func=None)

    def test(self, dl=None):
        if dl is None:
            dl = self.dl_manager.get_test_dl()

        # loss_meter = AverageMeter()
        # metric_meters = [AverageMeter() for i in range(5)]
        stages = [[] for i in range(5)]
        targets = []

        items = []
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(dl):
                img = data['img']
                # label = data[self.label]
                fixation = data['fixation']
                outs = self.model(img)

                # loss = 0
                targets.append(fixation[0].data.cpu().numpy())
                item = [data['name'].item()]
                for j, out in enumerate(outs):
                    out = torch.squeeze(out, dim=1)
                    stages[j].append(out[0].data.cpu().numpy())
                    cur_acc = self._cal_jud_acc(out[0].cpu(), fixation[0].cpu())
                    item.append(cur_acc)
                    # metric_meters[j].update(cur_acc)

                    # out = torch.log(out
                    # labels_sum = torch.sum(label.contiguous().view(label.size(0), -1), dim=1)
                    # label /= labels_sum.contiguous().view(*labels_sum.size(), 1, 1).expand_as(label)
                    # loss += self.criteration(out, label.to(self.device))
                    # loss_meter.update(loss.data.cpu().item())
                # break
                item.append(np.argmax(item[1:])+1)
                items.append(item)

            # self.writer.append(**dict(
            #     metric=[item.avg for item in metric_meters]
            # ))
            df = pd.DataFrame(items)
            df.to_csv(os.path.join(self.save_dir, 'score.csv'), index=False)
            print(self.save_dir)
        return torch.tensor(stages), torch.tensor(targets)

    def dynamic_evalute(self, valid_dl=None, test_dl=None):
        if valid_dl is None:
            valid_dl = self.dl_manager.get_test_dl(phase='valid')
        if test_dl is None:
            test_dl = self.dl_manager.get_test_dl(phase='test')
        val_pred, val_target = self.test(valid_dl)
        test_pred, test_target = self.test(test_dl)
        # val_pred nstage * nsample * w * h, tensor
        # val_target nsample*w*h, tensor

        # TODO add save predict answers

        nstages = 5
        for p in range(1, 40):
            print('*'*100)
            _p = torch.FloatTensor(1).fill_(p * 1. / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, nstages))
            probs /= probs.sum()
            acc_val, T = self.dynamic_eval_find_threshold(val_pred, val_target, probs)
            acc_test = self.dynamic_eval_with_threshold(test_pred, test_target, T)
            print('valid jud acc: {}, test jud acc: {}, thresh : {}'.format(acc_val, acc_test, T))
            print('*' * 100)

    def dynamic_eval_find_threshold(self, logits, targets, p):
        '''
        :param logits: N*M*H*W n_stage *sample
        :param targets:
        :param p:
        :return:
        '''
        preds = logits
        logits = torch.reshape(logits, (logits.shape[0], logits.shape[1], -1))  # N*M*(HW)
        logits = torch.mean(logits, dim=-1) # N*M*1
        logits = torch.reshape(logits, (logits.shape[0], logits.shape[1]))
        n_stage, n_sample = logits.size()

        _, sorted_idx = logits.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)  # 标记哪些样本已经被筛选了
        T = torch.Tensor(n_stage).fill_(1e8)  # threshold

        for k in range(n_stage-1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])

            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = logits[k][ori_idx]
                        break
            filtered.add_(logits[k].ge(T[k]).type_as(filtered)) # 所有threshold大于当前直，都质为1

        T[n_stage-1] = -1e8

        jud_acc = 0.
        for i in range(n_sample):
            for k in range(n_stage):
                if logits[k][i] >= T[k]:
                    jud_acc += self._cal_jud_acc(preds[k][i], targets[i])
                    break

        return jud_acc * 100. / n_sample, T

    def dynamic_eval_with_threshold(self, logits, targets, T):
        preds = logits
        logits = torch.reshape(logits, (logits.shape[0], logits.shape[1], -1))  # N*M*(HW)
        logits = torch.mean(logits, dim=-1)  # N*M*1
        logits = torch.reshape(logits, (logits.shape[0], logits.shape[1]))
        n_stage, n_sample = logits.size()

        jud_acc = 0.
        for i in range(n_sample):
            for k in range(n_stage):
                if logits[k][i] >= T[k]:
                    jud_acc += self._cal_jud_acc(preds[k][i], targets[i])
                    break

        return jud_acc * 100. / n_sample

    def _cal_jud_acc(self, out, label):
        out = out.data.numpy()
        label = label.data.numpy().astype(np.uint8)
        return self.metric(out, label)