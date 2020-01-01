from .basic_inferencer import BasicInferencer
from cframe.utils import *
import torch
from cframe.inferencer.tools.segment_postprocess import *
from cframe.inferencer.tools.fixation_postprocess import postprocess_predictions as postprocess_fixation

class Inferencer(BasicInferencer):
    def test(self, dl=None):
        if dl is None:
            dl = self.dl_manager.get_test_dl()
        # 记录总体的准确率
        # metric_meters = [AverageMeter() for i in range(len(self.metric_funcs))]

        items = []

        with torch.no_grad():
            self.model.eval()
            for i, data in tqdm(enumerate(dl)):
                img = data['img']
                out = self.model(img)

                # if not isinstance(out, list):
                #     out = [out]

                if 'save' in self.after_func.__name__:
                    self.after_func(self, out, data)
                else:
                    item, metrics = self.after_func(self, out, data)
                    # for i in range(len(metrics)):
                    #     metric_meters[i].update(metrics[i], img.shape[0])
                    items.append(item)
                    # break

        if self.save_dir is not None:
            if self.after_func.__name__ == 'multi_metrics':
                columns = ['name']
                columns.extend(self.metrics)
                df = pd.DataFrame(items, columns=columns)
                df.to_csv(os.path.join(self.save_dir, 'score.csv'), index=False)
                return df
            elif self.after_func.__name__ == 'multi_outputs':
                df = pd.DataFrame(items)
                df.to_csv(os.path.join(self.save_dir, 'score.csv'), index=False)
                return df
            elif 'save' in self.after_func.__name__:
                print('save predictions in {}'.format(self.save_dir))

    def predict(self, data_dict):
        img_path = data_dict['img']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape_r, shape_c, _ = img.shape
        img = self.dl_manager.img_transform(img)
        img = torch.stack([img], dim=0)
        with torch.no_grad():
            self.model.eval()
            out = self.model(img)

        out = out.data.cpu().numpy()
        if self.learner.task in ['sod' , 'segment' , 'Segment', 'SOD']:
            post_process = postprocess_segment
            out = post_process(out)
            return out[0]
        elif self.learner.task == 'fixation':
            post_process = postprocess_fixation
            out = out[0].squeeze()
            out = out[:, :, None]
            out = post_process(out, shape_r=shape_r, shape_c=shape_c)
            return out

        else:
            raise NotImplementedError

