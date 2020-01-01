import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from .basic_dataset import BasicDataset


class SegmentDataset(BasicDataset):
    def __init__(self, configer, phase,
                 img_transform, segment_transform=None, aug_transform=None):
        super(SegmentDataset, self).__init__(configer, phase)

        self.img_transform = img_transform
        self.segment_transform = segment_transform
        self.aug_transform = aug_transform

        self.imgs, self.segments = self.get_data_list()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = img.convert('RGB')
        img = np.array(img)

        segment = Image.open(self.segments[index])
        #  segment = segment.convert('L')
        segment = np.array(segment)

        if self.aug_transform is not None:
            img, segment = self.aug_transform(img, segment)

        img = np.array(img)
        segment = np.array(segment)
        img = self.img_transform(img)

        if self.segment_transform is not None:
            segment = self.segment_transform(segment)

        return dict(img=img,
                    segment=segment,
                    name=self.imgs[index].split('/')[-1].split('.')[0],
                    img_path=self.imgs[index],
                    segment_path=self.segments[index])

    def get_data_list(self):
        df = pd.read_csv(
            os.path.join(self.root_dir,
                         self.csv_dir,
                         self.data_name,
                         '{}.csv'.format(self.phase))
        )

        imgs = []
        segments = []

        for i in range(len(df)):
            imgs.append(os.path.join(self.root_dir,
                                     df.loc[i, 'img'])
                        )
            segments.append(os.path.join(self.root_dir,
                                       df.loc[i, self.label_name])
                          )
        return imgs, segments
