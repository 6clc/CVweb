from .basic_inferencer import BasicInferencer
from cframe.utils import *


class ClassificationInferencer(BasicInferencer):
    def __init__(self, model_manager, dl_manager, metric, label):
        super(ClassificationInferencer, self).__init__(model_manager=model_manager,
                                                       dl_manager=dl_manager,
                                                       metric=metric,
                                                       label=label)

    def test(self, dl=None):
        if dl is None:
            dl = self.dl_manager.get_test_dl()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()
        items = []

        for i, data in tqdm(enumerate(dl)):
            img = data['img']
            label = data[self.label]

            out = self.model(img)

            pred = out.data.cpu().numpy()[0]
            pred = np.argmax(pred, axis=-1)

            loss = self.criteration(out, label.to(self.device))
            loss_meter.update(loss.data.cpu().item())

            name = data['name'][0]
            item = [name]
            items.extend(out.data.cpu().numpy()[0])
            cur_metric = self.metric(out.data.cpu().numpy(), label.data.cpu().numpy())
            item.append(cur_metric)
            metric_meter.update(cur_metric)
            item.extend([pred, label])
            items.append(item)

        items.append(['mean_metric'].extend([metric_meter.avg for metric_meter in metric_meters]))
        if self.save_dir is not None:
            df = pd.DataFrame(items, columns=['name'].extend([metric.__name__ for metric in self.metrics]).extend(['pred', 'label']))
            df.to_csv(os.path.join(self.save_dir, 'score.csv'))
