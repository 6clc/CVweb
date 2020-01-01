import numpy as np
import copy
import torch
from torch import tensor


def tensor_iou(input, targs, classes=2):
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n, -1)
    targs = targs.view(n, -1)

    ious = torch.zeros([n, classes])

    for c in range(classes):
        input_ = (input == c)
        targs_ = (targs == c)
        intersect_ = (input_ * targs_).sum(dim=1).float()
        union_ = (input_ + targs_).sum(dim=1).float()
        ious[:, c] = intersect_ / (union_ - intersect_ + 1.)
    res = ious.sum(dim=0) / n

    res = res.sum() / (classes)
    return tensor(res)


def dice_score(prob, truth, threshold=0.5):
    num = prob.shape[0]
    # prob = np.squeeze(prob)

    prob = prob > threshold
    truth = truth > 0.5

    prob = prob.reshape(num, -1)
    truth = truth.reshape(num, -1)
    intersection = (prob * truth)

    score = (2. * (intersection.sum(1) + 1.)) / (prob.sum(1) + truth.sum(1) + 2.)
    score[score >= 1] = 1
    score = score.sum() / num

    return score


def acc(preds, target):
    preds = np.transpose(preds, [0, 2, 3, 1])
    preds = np.argmax(preds, axis=-1)
    n = preds.shape[0]
    preds = preds.reshape(n, -1)
    target = target.reshape(n, -1)
    return np.sum(preds == target).astype(np.float).mean()


def f_score(preds, target, beta=0.3, eps=1e-6):
    beta2 = beta**2
    preds = np.transpose(preds, [0, 2, 3, 1])
    preds = np.argmax(preds, axis=-1)
    n = preds.shape[0]
    preds = preds.reshape(n, -1)
    target = target.reshape(n, -1)
    TP = np.sum(preds*target, axis=1)
    prec = TP / (np.sum(preds)+eps)
    rec = TP / (np.sum(target, axis=1)+eps)
    res = (prec*rec) / (prec*beta2 + rec + eps)*(1+beta2)
    return res.mean()


def batch_iou(preds, targets, thresh=0.5):
    '''
    for np.argmax(preds) to get answer
    :param preds: N * C * H * W, ndarray
    :param targets: N * H * W, ndarray
    :return:
    '''
    preds = copy.deepcopy(preds)
    targets = copy.deepcopy(targets)
    if not isinstance(preds, np.ndarray):
        preds = preds.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
    n_classes = preds.shape[1]
    confusion_matrix = np.zeros((n_classes, n_classes))

    preds = np.transpose(preds, [0, 2, 3, 1])
    preds = np.argmax(preds, axis=-1)

    for lt, lp in zip(targets, preds):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten(), n_classes)
    return get_scores(confusion_matrix, n_classes)[3]

def iou(pred, target):
    preds = copy.deepcopy(pred)
    targets = copy.deepcopy(target)
    if not isinstance(preds, np.ndarray):
        preds = preds.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
    preds = np.expand_dims(preds, axis=0)
    targets = np.expand_dims(target, axis=0)
    n_classes = preds.shape[1]
    confusion_matrix = np.zeros((n_classes, n_classes))

    preds = np.transpose(preds, [0, 2, 3, 1])
    preds = np.argmax(preds, axis=-1)

    for lt, lp in zip(targets, preds):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten(), n_classes)
    return get_scores(confusion_matrix, n_classes)[3]

def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)

    return hist


def get_scores(confusion_matrix, n_classes):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(n_classes), iu))

        return acc, acc_cls, fwavacc, mean_iu, cls_iu
