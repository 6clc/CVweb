from PIL import Image
from torch.utils.data import DataLoader
import cv2
from cframe.dataloader.dataset import FixationDataset
from cframe.dataloader.tools import standard_transform


class FixationDataloaderManager(object):
    def __init__(self, config):
        self.config = config

        self.img_transform = None
        self.saliency_transform = None
        self.fixation_transform = None

        self.img_trans_list = None
        self.saliency_trans_list = None
        self.fixation_trans_list = None

        self.pred_img_transfom = None
        self.pred_saliency_transform = None
        self.pred_fixation_tranform = None

        self.aug_transform = None

        self._init()

    def _init(self):
        size = self.config['resize']

        if 'img_resize' in self.config['data_info']:
            img_resize = self.config['data_info']['img_resize']
            print('img will be resized to {}'.format(img_resize))
        else:
            print('img resize will not bte sized')
            img_resize = False

        if 'gt_resize' in self.config['data_info']:
            gt_resize = self.config['data_info']['gt_resize']
            size = gt_resize
            print('gt will be resized to {}'.format(gt_resize))
        else:
            print('gt will not be resized')
            gt_resize = False

        self.img_trans_list = [standard_transform.PIL_ReSize(img_resize),
                               standard_transform.ToTensor(d255=True),
                               standard_transform.Normalize(**self.config['normalize'])]
        self.saliency_trans_list = [standard_transform.ReSize(size, interpolation=cv2.INTER_LINEAR),
                                    standard_transform.ToTensor(d255=False)]
        self.fixation_trans_list = [standard_transform.ReSize(size, interpolation=cv2.INTER_NEAREST),
                                    standard_transform.ToLabel(),
                                    standard_transform.ReLabel(255, 1)]
        if img_resize:
            self.img_transform = standard_transform.Compose(self.img_trans_list)
        else:
            self.img_transform = standard_transform.Compose(self.img_trans_list[1:])

        if gt_resize:
            self.saliency_transform = standard_transform.Compose(self.saliency_trans_list[1:])
            self.fixation_transform = standard_transform.Compose(self.fixation_trans_list[1:])
        else:
            self.saliency_transform = standard_transform.Compose(self.saliency_trans_list[1:])
            self.fixation_transform = standard_transform.Compose(self.fixation_trans_list[1:])

    def get_train_dl(self, phase='train'):
        train_dl_dict = self.config[phase]

        return DataLoader(FixationDataset(self.config, phase=phase,
                                          img_transform=self.img_transform,
                                          saliency_transform=self.saliency_transform,
                                          fixation_transform=self.fixation_transform,
                                          aug_transform=self.aug_transform),
                          shuffle=True,
                          batch_size=train_dl_dict['batch_size'],
                          num_workers=train_dl_dict['num_workers'],
                          drop_last=True)

    def get_valid_dl(self, phase='valid'):
        valid_dl_dict = self.config[phase]
        return DataLoader(FixationDataset(self.config, phase=phase,
                                          img_transform=self.img_transform,
                                          saliency_transform=self.saliency_transform,
                                          fixation_transform=self.fixation_transform),
                          shuffle=False,
                          batch_size=valid_dl_dict['batch_size'],
                          num_workers=valid_dl_dict['num_workers']
                          )

    def get_test_dl(self, phase='test'):
        return DataLoader(FixationDataset(self.config, phase=phase,
                                          img_transform=self.img_transform,
                                          saliency_transform=self.saliency_transform,
                                          fixation_transform=self.fixation_transform),
                          shuffle=False,
                          batch_size=1,
                          num_workers=0
                          )


if __name__ == '__main__':
    from cframe.dataloader.data_config import Dataconfig

    data_config = Dataconfig.get_data_config('DUT')
    dl_manager = FixationDataloaderManager(data_config)
    valid_dl = dl_manager.get_train_dl()
    for i, data in enumerate(valid_dl):
        print(data['img'].shape, data['fixation'].shape, data['saliency'].shape)
        break
