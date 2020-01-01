import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FixationDataset(Dataset):
    def __init__(self, configer, phase, img_transform,
                 saliency_transform=None, fixation_transform=None, aug_transform=None):
        self.configer = configer
        self.phase = phase
        self.root_dir = self.configer['root_dir']
        self.dataset_dir = self.configer['dataset_dir']
        self.csv_dir = self.configer['csv_dir']
        self.data_name = self.configer['data_name']

        self.img_transform = img_transform
        self.saliency_transform = saliency_transform
        self.fixation_transform = fixation_transform
        self.aug_transform = aug_transform

        self.imgs, self.saliencys, self.fixations = self.get_data_list()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        saliency = cv2.imread(self.saliencys[index], cv2.IMREAD_GRAYSCALE)
        fixation = cv2.imread(self.fixations[index], cv2.IMREAD_GRAYSCALE)

        if self.aug_transform is not None:
            img, label, fixation = self.aug_transform(img, saliency, fixation)

        img = self.img_transform(img)
        saliency = self.saliency_transform(saliency)
        fixation = self.fixation_transform(fixation)

        return dict(img=img,
                    saliency=saliency,
                    fixation=fixation,
                    name=self.imgs[index].split('/')[-1])

    def get_data_list(self):
        df = pd.read_csv(
            os.path.join(self.root_dir,
                         self.csv_dir,
                         self.data_name,
                         '{}.csv'.format(self.phase))
        )

        imgs = []
        saliencys = []
        fixations = []

        for i in range(len(df)):
            imgs.append(os.path.join(self.root_dir, df.loc[i, 'img']))
            saliencys.append(os.path.join(self.root_dir, df.loc[i, 'saliency']))
            fixations.append(os.path.join(self.root_dir, df.loc[i, 'fixation']))
        return imgs, saliencys, fixations
