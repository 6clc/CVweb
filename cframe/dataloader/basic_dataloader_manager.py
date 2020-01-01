from cframe.dataloader.tools import standard_transform


class BasicDataloaderManager(object):
    def __init__(self, configer):
        self.configer = configer
        self.img_transform = None
        self.img_trans_list = None
        self.pred_img_transfom = None
        self.aug_transform = None

        self.img_transform = None
        self.saliency_transform = None
        self.fixation_transform = None

    def get_train_dl(self, phase='train'):
        raise NotImplementedError

    def get_valid_dl(self, phase='valid'):
        raise NotImplementedError

    def get_test_dl(self, phase='test'):
        raise NotImplementedError


