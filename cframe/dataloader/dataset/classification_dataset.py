from cframe.utils import *
from .basic_dataset import BasicDataset


class ClassificationDataset(BasicDataset):
    def __init__(self, configer, phase, img_transform, aug_transform=None, change_label=None):
        super(ClassificationDataset, self).__init__(configer, phase)

        self.img_transform = img_transform
        self.aug_transform = aug_transform
        self.change_label = change_label

        self.imgs, self.labels = self.get_data_list()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = np.array(img)
        label = self.labels[index]

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        img = self.img_transform(img)

        if self.change_label is not None:
            label = self.change_label(label)

        return dict(img=img,
                    label=label,
                    name=self.imgs[index].split('\\')[-1])

    def get_data_list(self):
        df = pd.read_csv(
            os.path.join(self.root_dir,
                         self.csv_dir,
                         self.data_name,
                         '{}.csv'.format(self.phase))
        )

        imgs = []
        labels = []

        for i in range(len(df)):
            imgs.append(os.path.join(self.root_dir,
                                     df.loc[i, 'img'])
                        )
            labels.append(df.loc[i, self.label_name])
        return imgs, labels
