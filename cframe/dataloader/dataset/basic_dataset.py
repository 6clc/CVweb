from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, configer, phase):
        self.configer = configer
        self.phase = phase
        self.root_dir = self.configer['root_dir']
        self.dataset_dir = self.configer['dataset_dir']
        self.csv_dir = self.configer['csv_dir']
        self.data_name = self.configer['data_name']
        self.label_name = self.configer['label_name']

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_data_list(self):
        raise NotImplementedError