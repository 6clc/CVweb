import numpy as np
from .fixation_postprocess import postprocess_predictions
import os
import cv2


def solve_classification(inferencer, **data):
    name = data['name'][0]
    out = data['out'].data.cpu().numpy()[0]
    label = data[inferencer.label].data.cpu().numpy()[0]
    pred = np.argmax(out, axis=-1)
    item = [name, out, pred, label]

    cur_metric = inferencer.metric(out, label)
    return item, cur_metric


def multi_metrics(inferencer, out, data):
    name = data['name'][0]
    out = out.data.cpu().numpy()[0].squeeze()
    label = data['fixation'].data.cpu().numpy()[0]
    # print(out.shape, label.shape)

    item = [name]
    metric_items = []
    for metric in inferencer.metric_funcs:
        metric_items.append(metric(out, label))
    item.extend(metric_items)
    return item, metric_items

def multi_outputs(inferencer, outs, data):
    name = data['name'][0]
    label = data['fixation'].data.cpu().numpy()[0]
    # print(out.shape, label.shape)

    item = [name]
    metric_items = []
    for out in outs:
        out = out.data.cpu().numpy()[0].squeeze()
        metric_items.append(inferencer.metric_funcs[-1](out, label))
    item.extend(metric_items)
    return item, metric_items

def save_fixations(inferencer, out, data):
    name = data['name'][0]
    out = out.data.cpu().numpy()[0].squeeze()
    out = out[:, :, None]
    shape_r, shape_c = inferencer.model_manager.model_config['config']['gt_shape']
    out = postprocess_predictions(out, shape_r=shape_r, shape_c=shape_c)
    dst_dir = os.path.join(inferencer.save_dir, name)
    cv2.imwrite(dst_dir, out.astype(np.uint8))
