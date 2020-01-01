from torch import optim
from torch.optim.lr_scheduler import *
from cframe.dataloader import *
from cframe.learner.tools import CyclicLR
from cframe.utils import *
from cframe.utils.fastprogress import *
from .basic_learner import BasicLearner


class MultiOutLearner(BasicLearner):
    def __init__(self, model_manager, dl_manager, metric, optimizer, scheduler, label='label'):
        super(MultiOutLearner, self).__init__(model_manager, dl_manager, metric, optimizer, scheduler, label)

    def train(self, num_epoches, log_nth=None, show_batch_loss=False):
        if log_nth is None:
            log_nth = len(self.train_dl)
        self.writer = SummaryWriter(*['train_loss', 'valid_loss', 'batch_loss', 'metric', 'lr'])
        iter_per_epoch = len(self.train_dl)
        n_iterations = num_epoches * iter_per_epoch

        mb = master_bar(range(num_epoches))
        mb.write(['train_loss', 'valid_loss', 'metric', 'lr'], table=True)
        mb.names = ['train_loss', 'valid_loss', 'batch_loss']

        train_loss_meter = AverageMeter()
        for epoch in mb:
            for i, data in enumerate(progress_bar(self.train_dl, parent=mb)):
                it = epoch * iter_per_epoch + i + 1
                # print(i, data)
                img = data['img']
                label = data[self.label]
                label = label.unsqueeze(dim=1).to(self.device)

                # outs, s = self.model(img)
                outs = self.model(img)

                if isinstance(outs, dict):
                    outs = outs.values()
                if not isinstance(outs, list):
                    outs = [outs]
                loss = 0
                for stage, out in enumerate(outs):
                    loss += self.criteration(out, label)
                    # B, C, H, W = out.shape
                    # for batch_idx in range(B):
                    #     loss += s[batch_idx][stage]*self.criteration(
                    #         torch.unsqueeze(out[batch_idx], dim=1),
                    #         torch.unsqueeze(label[batch_idx], dim=1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if isinstance(self.scheduler, CyclicLR):
                    self.scheduler.step()
                train_loss_meter.update(loss.data.cpu().item())
                self.writer.append(**dict(
                    batch_loss=loss.data.cpu().item(),
                    lr=self.optimizer.param_groups[0]['lr']
                ))

                if it % log_nth == 0:
                    self.writer.append(**dict(
                        train_loss=train_loss_meter.avg,
                    ))
                    train_loss_meter.reset()
                    self.model.eval()
                    with torch.no_grad():
                        valid_loss = self._train_validate(mb, epoch, num_epoches, log_nth, show_batch_loss)
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(valid_loss)
                    else:
                        self.scheduler.step()
                    self.model.train()
        self.save()
        self.writer.save(path=self.summary_writer_dir)


    def _train_validate(self, mb, epoch, num_epoches, log_nth, show_batch_loss):
        valid_loss_meter = AverageMeter()
        valid_acc_meters = [AverageMeter() for i in range(4)]
        for i, data in enumerate(self.valid_dl):
            img = data['img']
            label = data[self.label].unsqueeze(dim=-1)
            fixation = data['fixation']
            # outs, s = self.model(img)
            outs = self.model(img)

            loss = 0
            for j, out in enumerate(outs):
                cur_acc = self.metric(out, fixation)
                # cur_acc = -1.
                valid_acc_meters[j].update(cur_acc)

                # out = torch.log(out)
                # labels_sum = torch.sum(label.contiguous().view(label.size(0), -1), dim=1)
                # label /= labels_sum.contiguous().view(*labels_sum.size(), 1, 1).expand_as(label)
                loss += self.criteration(out, label.to(self.device))
            valid_loss_meter.update(loss.data.cpu().item())

        self.writer.append(**dict(
            valid_loss=valid_loss_meter.avg,
            metric=[item.avg for item in valid_acc_meters]
        ))
        train_loss = self.writer.get('train_loss')
        valid_loss = self.writer.get('valid_loss')
        batch_loss = self.writer.get('batch_loss')
        lr = self.writer.get('lr')
        metric = self.writer.get('metric')

        # print(train_loss, valid_loss)
        mb.write(['%2.2f' % round(train_loss[-1], 2),
                  '%2.2f' % round(valid_loss[-1], 2),
                  ' | '.join(['%2.2f%%' % round(item*100, 2) for item in metric[-1]]),
                  '%.0e' % lr[-1]
                  ], table=True)
        log_iterations = [(item) * log_nth for item in range(1, len(valid_loss) + 1)]
        batch_iterations = [item for item in range(1, len(lr) + 1)]

        if show_batch_loss:
            loss_graph_node = [[log_iterations, train_loss],
                               [log_iterations, valid_loss], [batch_iterations, batch_loss]]
        else:
            loss_graph_node = [[log_iterations, train_loss], [log_iterations, valid_loss]]
        lr_graph_node = [batch_iterations, lr]
        mb.update_graph(loss_graphs=loss_graph_node, lr_graphs=lr_graph_node,
                        x_bounds=[1, len(self.train_dl) * (num_epoches+1) + 1],
                        y_bounds=[min(train_loss), max(valid_loss)],
                        lr_bounds=[0, max(lr) + (1e-10)])

        self.best_para = self.get_para()
        return valid_loss_meter.avg