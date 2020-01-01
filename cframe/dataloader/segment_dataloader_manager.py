from PIL import Image
from torch.utils.data import DataLoader

from cframe.dataloader.basic_dataloader_manager import BasicDataloaderManager
from cframe.dataloader.dataset import SegmentDataset
from cframe.dataloader.tools import DefaultSegAug
from cframe.dataloader.tools import standard_transform


class SegmentDataloaderManager(BasicDataloaderManager):
    def __init__(self, configer):
        super(SegmentDataloaderManager, self).__init__(configer)
        self.config = configer
        self.aug_transform = DefaultSegAug(configer)
        self._init()

    def _init(self):
        size = self.configer['resize']
        self.img_trans_list = standard_transform.get_image_transform_list(size, self.configer['normalize'])
        self.segment_trans_list = standard_transform.get_mask_transform_list(size)

        self.img_transform = standard_transform.Compose(self.img_trans_list)
        self.segment_transform = standard_transform.Compose(self.segment_trans_list)

        self.pred_img_transform = standard_transform.Compose(self.img_trans_list[1:])
        self.pred_segment_transform = standard_transform.Compose(self.segment_trans_list[1:])

    def get_train_dl(self, phase='train'):
        train_dl_dict = self.configer[phase]
        return DataLoader(SegmentDataset(self.configer, phase=phase,
                                         img_transform=self.img_transform,
                                         segment_transform=self.segment_transform,
                                         aug_transform=self.aug_transform),
                          shuffle=True,
                          batch_size=train_dl_dict['batch_size'],
                          num_workers=train_dl_dict['num_workers'],
                          drop_last=True)

    def get_dl(self, phase='valid'):
        valid_dl_dict = self.configer[phase]
        if phase == 'train':
            aug_transform = self.aug_transform
        else:
            aug_transform = None
        return DataLoader(SegmentDataset(self.configer, phase=phase,
                                         img_transform=self.img_trans_list[0],
                                         segment_transform=self.segment_trans_list[0],
                                         aug_transform=aug_transform),
                          shuffle=False,
                          batch_size=valid_dl_dict['batch_size'],
                          num_workers=valid_dl_dict['num_workers']
                          )

    def get_valid_dl(self, phase='valid'):
        valid_dl_dict = self.configer[phase]
        return DataLoader(SegmentDataset(self.configer, phase=phase,
                                         img_transform=self.img_transform,
                                         segment_transform=self.segment_transform,
                                         aug_transform=None),
                          shuffle=False,
                          batch_size=valid_dl_dict['batch_size'],
                          num_workers=valid_dl_dict['num_workers']
                          )

    def get_test_dl(self, phase='test'):
        return DataLoader(SegmentDataset(self.configer, phase=phase,
                                         img_transform=self.pred_img_transform,
                                         segment_transform=self.pred_segment_transform),
                          shuffle=False,
                          batch_size=1,
                          num_workers=0
                          )


if __name__ == '__main__':
    from cframe.dataloader.data_configer import DataConfiger
    data_config = DataConfiger.get_data_config('CamVid')
    dl_manager = SegmentDataloaderManager(data_config)
    valid_dl = dl_manager.get_train_dl()
    for i, data in enumerate(valid_dl):
        print(data['img'].shape, data['segment'].shape)
        break
