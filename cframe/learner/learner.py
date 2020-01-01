from abc import ABC

from torch.optim.lr_scheduler import *
from cframe.dataloader import *
from cframe.inferencer.tools.classification_postprocess import postprocess_classification
from cframe.learner.tools import CyclicLR
from cframe.utils import *
from cframe.utils.fastprogress import *
from .basic_learner import BasicLearner
from random import randint
import time
from cframe.inferencer.tools.segment_postprocess import *

class Learner(BasicLearner):

    def train(self, num_epoches, log_nth=None, show_batch_loss=False):
        if log_nth is None:
            log_nth = len(self.train_dl)

        writer_name = ['train_loss', 'valid_loss', 'batch_loss', 'lr']
        column_name = ['train_loss', 'valid_loss', 'lr']
        for i in range(self.num_output):
            writer_name.append(self.metric_names+str(i+1))
            column_name.append(self.metric_names+str(i+1))

        self.writer = SummaryWriter(*writer_name)
        mb = master_bar(range(num_epoches))
        mb.write(column_name, table=True)
        mb.names = ['train_loss', 'valid_loss', 'batch_loss']

        iter_per_epoch = len(self.train_dl)
        # n_iterations = num_epoches * iter_per_epoch

        train_loss_meter = AverageMeter()
        self.running_score['best_metric'] = 0.0

        for epoch in mb:
            for i, data in enumerate(progress_bar(self.train_dl, parent=mb)):
                it = epoch * iter_per_epoch + i + 1

                img = data['img']

                loss_dict = dict()
                if self.task == 'fixation':
                    for label_name in self.label:
                        loss_dict[label_name] = data[label_name].float().cuda()
                else:
                    for label_name in self.label:
                        loss_dict[label_name] = data[label_name].cuda()

                outs = self.model(img)
                if isinstance(outs, dict):
                    outs = outs.values()
                if not isinstance(outs, list):
                    outs = [outs]
                # print(label.shape, out.shape)
                loss = 0
                for stage, out in enumerate(outs):
                    # print(out.shape, label.shape)
                    out = torch.squeeze(out, dim=1)
                    # return self.criteration(out, label.to(out.device))
                    loss_dict['pred'] = out
                    loss += self.criteration(loss_dict)
                # print(loss.item())

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
        valid_metric_meters = []
        for i in range(self.num_output):
            valid_metric_meters.append([AverageMeter() for j in range(len(self.metrics))])

        for i, data in enumerate(progress_bar(self.valid_dl, parent=mb)):
            img = data['img']

            loss_dict = dict()
            if self.task == 'fixation':
                for label_name in self.label:
                    loss_dict[label_name] = data[label_name].float().cuda()
            else:
                for label_name in self.label:
                    loss_dict[label_name] = data[label_name].cuda()

            model_start = time.time()
            outs = self.model(img)
            if isinstance(outs, dict):
                outs = outs.values()
            if not isinstance(outs, list):
                outs = [outs]
            batch = img.shape[0]
            model_end = time.time()

            loss = 0
            start = time.time()
            for j, out in enumerate(outs):
                for k in range(len(self.metrics)):
                    cur_acc = self.batch_metric_funcs[k](out, loss_dict[label_name])
                    valid_metric_meters[j][k].update(cur_acc)
                out = torch.squeeze(out, dim=1)
                loss_dict['pred'] = out
                loss += self.criteration(loss_dict)
            valid_loss_meter.update(loss.data.cpu().item())
            end = time.time()

            # print('time %.2f model time %.2f' % (end-start, model_end-model_start))

        writer_log = dict(valid_loss=valid_loss_meter.avg)
        for i in range(self.num_output):
            writer_log[self.metric_names+str(i+1)] = [item.avg for item in valid_metric_meters[i]]
        self.writer.append(**writer_log)

        if valid_metric_meters[-1][-1].avg > self.running_score['best_metric']:
            self.running_score['best_metric'] = valid_metric_meters[-1][-1].avg
            self.best_para = self.get_para()

        train_loss = self.writer.get('train_loss')
        valid_loss = self.writer.get('valid_loss')
        batch_loss = self.writer.get('batch_loss')
        lr = self.writer.get('lr')

        # print(train_loss, valid_loss)
        cur_metircs = ['%2.2f' % round(train_loss[-1], 2),
                  '%2.2f' % round(valid_loss[-1], 2),
                  '%.0e' % lr[-1]]
        for i in range(self.num_output):
            cur_metircs.append(' | '.join(['%.2f' % round(item, 2)
                                           for item in self.writer.get(self.metric_names+str(i+1))[-1]]))
        mb.write(cur_metircs, table=True)
        log_iterations = [(item) * log_nth for item in range(1, len(valid_loss) + 1)]
        batch_iterations = [item for item in range(1, len(lr) + 1)]

        if show_batch_loss:
            loss_graph_node = [
                [log_iterations, train_loss],
                [log_iterations, valid_loss],
                [batch_iterations, batch_loss]]
        else:
            loss_graph_node = [[log_iterations, train_loss], [log_iterations, valid_loss]]
        lr_graph_node = [batch_iterations, lr]

        mb.update_graph(loss_graphs=loss_graph_node, lr_graphs=lr_graph_node,
                        x_bounds=[1, len(self.train_dl) * (num_epoches + 1) + 1],
                        y_bounds=[min(train_loss), max(valid_loss)],
                        lr_bounds=[0, max(lr) + (1e-10)])
        return valid_loss_meter.avg

    def show_any_batch_results(self, dl=None):
        if dl is None:
            dl = self.valid_dl
        dl_iter = iter(dl)
        data = next(dl_iter)

        imgs = data['img']
        with torch.no_grad():
            self.model.eval()
            outs = self.model(imgs)
            self.model.train()

        if isinstance(outs, dict):
            outs = outs.values()
        if not isinstance(outs, list):
            outs = [outs]
        # outs = outs[-1]

        denormalizer = DeNormalize(**self.dl_manager.configer['normalize'])
        for i in range(imgs.size(0)):
            imgs[i] = denormalizer(imgs[i])
            imgs[i] = imgs[i]*255
        imgs = imgs.data.numpy().astype(np.uint8)
        imgs = np.transpose(imgs, [0, 2, 3, 1])

        labels = []
        self.post_process = None
        for label_name in self.label:
            labels.append(data[label_name].data.numpy().squeeze().astype(np.uint8))
        if self.task in ['Segment', 'segment', 'SOD']:
            self.post_process = postprocess_segment
        if self.task in ['sense_number', 'classification']:
            self.post_process = postprocess_classification

        for i in range(len(outs)):
            if self.post_process is not None:
                outs[i] = outs[i].data.cpu().numpy()
                outs[i] = self.post_process(outs[i])
            else:
                outs[i] = outs[i].data.cpu().numpy().squeeze()

        if self.task in ['Segment', 'segment', 'SOD']:
            self.show_xyzs(imgs, labels, outs)
        if self.task in ['sense_number', 'classification']:
            self.show_xs(imgs, labels, outs)



        return dict(imgs=imgs, labels=labels, outs=outs)

    def show_xyzs(self,
            imgs, labels, outs,
            imgsize: int = 4, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = 'Input / Prediction / Target'
        batch_size = imgs.shape[0]
        col = 1 + len(labels) + len(outs)
        figsize = (imgsize*col, imgsize*batch_size)
        fig, axs = plt.subplots(batch_size, col, figsize=figsize)
        [axi.set_axis_off() for axi in axs.ravel()]

        for i in range(batch_size):
            axs[i, 0].imshow(imgs[i, :])
            if i == 0:
                axs[i, 0].set_title('img')

            st = 1
            for j in range(len(labels)):
                axs[i, st + j].imshow(labels[j][i, :])
                if i == 0:
                    axs[i, st + j].set_title(self.label[j])

            st += len(labels)
            for j in range(len(labels)):
                axs[i, st+j].imshow(outs[j][i, :])
                if i==0:
                    axs[i, st+j].set_title('prediction')
                # items = []
                # for k, metric_name in enumerate(self.metrics):
                #     items.append(round(self.metric_funcs[k](outs[j][i, :],
                #                                             labels[-1][i, :]), 2))


                # the_table1 = axs[i, st+j].table(cellText=[items, ], colLabels=self.metrics,
                #                   cellLoc='center', edges='open')
                #
                # the_table1.auto_set_font_size(False)
                # the_table1.set_fontsize(10)
                # the_table1.scale(1, 1)
        fig.savefig(os.path.join(self.summary_writer_dir, 'show_any_batch.png'))
            # axs[i, col-1].text(0.2, float(k)/len(self.metrics),
            #                    '{} = {}'.format(metric_name, round(cur_metric, 2)), size=15)
            # # axs[i, :].axis('off')
        # fig.suptitle(title)                axs[i, 3+j].table(cellText=[item], loc='right', rowLabels

    def show_xs(self,
                  imgs, labels, outs,
                  imgsize: int = 4, **kwargs):

        batch_size = imgs.shape[0]
        col = 1
        figsize = (imgsize * col, imgsize * batch_size)
        fig, axs = plt.subplots(batch_size, col, figsize=figsize)
        [axi.set_axis_off() for axi in axs.ravel()]

        for i in range(batch_size):
            axs[i].imshow(imgs[i, :])
            axs[i].set_title('gt {}, pred {}'.format(labels[0][i], outs[0][i]))

        fig.savefig(os.path.join(self.summary_writer_dir, 'show_any_batch.png'))





