from torch.utils.data import DataLoader
from cframe.dataloader.basic_dataloader_manager import BasicDataloaderManager
from cframe.dataloader.dataset import ClassificationDataset
from cframe.dataloader.tools import DefaultClassAug
from cframe.dataloader.tools import standard_transform


class ClassificationDataloaderManager(BasicDataloaderManager):
    def __init__(self, configer, change_label=None):
        super(ClassificationDataloaderManager, self).__init__(configer)
        self.config = configer
        self.aug_transform = DefaultClassAug(configer)

        self.img_trans_list = standard_transform.get_image_transform_list(self.configer['resize'],
                                                                          self.configer['normalize'])
        self.img_transform = standard_transform.Compose(self.img_trans_list)
        self.pred_img_transform = standard_transform.Compose(self.img_trans_list[1:])
        self.change_label = change_label

    def get_train_dl(self, phase='train'):
        train_dl_dict = self.configer[phase]
        return DataLoader(ClassificationDataset(self.configer, phase=phase,
                                         img_transform=self.img_transform,
                                         aug_transform=self.aug_transform,
                                                change_label=self.change_label),
                          shuffle=True,
                          batch_size=train_dl_dict['batch_size'],
                          num_workers=train_dl_dict['num_workers'],
                          drop_last=True)

    def get_valid_dl(self, phase='valid'):
        valid_dl_dict = self.configer[phase]
        return DataLoader(ClassificationDataset(self.configer, phase=phase,
                                         img_transform=self.img_transform,
                                         aug_transform=None,
                                                change_label=self.change_label),
                          shuffle=True,
                          batch_size=valid_dl_dict['batch_size'],
                          num_workers=valid_dl_dict['num_workers'],
                          drop_last=False)

    def get_test_dl(self, phase='test'):
        return DataLoader(ClassificationDataset(self.configer, phase=phase,
                                         img_transform=self.pred_img_transform,
                                                change_label=self.change_label),
                          shuffle=False,
                          batch_size=1,
                          num_workers=0
                          )




if __name__ == '__main__':
    from cframe.dataloader.data_configer import DataConfiger

    data_config = DataConfiger.get_data_config('leaf')
    dl_manager = ClassificationDataloaderManager(data_config)
    valid_dl = dl_manager.get_train_dl()
    for i, data in enumerate(valid_dl):
        print(data['img'].shape)
        break

