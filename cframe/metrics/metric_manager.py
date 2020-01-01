from .visual_attention_metrics import *
from .visual_segment_metrics import *

METRIC_DICT = dict(
    auc_jud=auc_jud,
    kld=KLdiv,
    nss=NSS,
    auc_shuff=auc_shuff,
    cc=cc,
    iou=iou
)

BATCH_METRIC_DICT = dict(
    auc_jud=batch_auc_jud,
    kld=batch_kldiv,
    nss=batch_nss,
    auc_shuff=batch_auc_shuff,
    cc=batch_cc,
    iou=batch_iou
)


def get_metric_fun(name, batch=True):
    if batch:
        return BATCH_METRIC_DICT[name]
    else:
        return METRIC_DICT[name]


def get_metric_funcs(names):
    metric_funcs = []
    batch_metric_funcs = []
    for name in names:
        metric_funcs.append(METRIC_DICT[name])
        batch_metric_funcs.append(BATCH_METRIC_DICT[name])
    return metric_funcs, batch_metric_funcs


def get_metirc_name():
    return METRIC_DICT.keys()
