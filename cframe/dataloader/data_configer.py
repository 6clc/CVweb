DATA_INFO = dict(
    LumbarSpinal=dict(dir='LumbarSpinal',
                      columns=['img', 'segment'],
                      n_classes=4),
    DUT=dict(dir='DUT-OMRON',
             columns=['img', 'fixation', 'saliency']),

    MSRA_B=dict(dir='MSRA_B', n_classes=2,
                columns=['img', 'saliency']),

    SOC6k=dict(dir='SOC6K', n_classes=2,
               columns=['img', 'instance', 'saliency', 'sense_number']),
    SALICON=dict(dir='SALICON',
                 img_resize=(240, 320),
                 columns=['img', 'saliency', 'fixation']),
    InstanceSaliency1000=dict(dir='InstanceSaliency1000', n_classes=2,
                              columns=['img', 'instance', 'saliency']),
    ECSSD=dict(dir='ECSSD', n_classes=2),
    MSRA_B_InstanceSaliency1000=dict(dir='MSRA_B_InstanceSaliency1000', n_classes=2,
                                     columns=['img', 'instance', 'saliency'])
)

DATA_ROOT = dict(
    root_dir='/run/media/hanshan/data2/Data',
    dataset_dir='DataSets',
    csv_dir='CSVs',
    data_name=None,
    data_info=None,
    label_name='segment',
    normalize=dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    resize=(224, 224),
    train=dict(
        batch_size=32,
        num_workers=8
    ),
    valid=dict(
        batch_size=32,
        num_workers=8,
    )
)


class DataConfiger(object):
    @classmethod
    def set_data_root_dir(cls, data_root_dir):
        DATA_ROOT['root_dir'] = data_root_dir

    @classmethod
    def get_all_data_name(cls):
        return DATA_INFO.keys()

    @classmethod
    def get_data_config(cls, data_name):
        new_config = DATA_ROOT
        new_config['data_info'] = DATA_INFO[data_name]
        new_config['data_name'] = DATA_INFO[data_name]['dir']
        return new_config
