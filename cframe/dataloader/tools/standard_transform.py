import numpy as np
import torch
from PIL import Image
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ReSize(object):
    def __init__(self, shape, interpolation=cv2.INTER_LINEAR):
        self.shape = shape
        self.interpolation = interpolation

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        min_side = self.shape[0]
        h = img.shape[0]
        w = img.shape[1]
        scale = max(w, h) / float(min_side)
        new_w, new_h = int(w/scale), int(h/scale)
        resize_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

        if new_w % 2 != 0 and new_h % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                        min_side - new_w) / 2
        elif new_h % 2 != 0 and new_w % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                        min_side - new_w) / 2
        elif new_h % 2 == 0 and new_w % 2 == 0:
            top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                        min_side - new_w) / 2
        else:
            top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                        min_side - new_w) / 2
        pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])  # 从图像边界向上,下,左,右扩的像素数目
        return pad_img


class PIL_ReSize(object):
    def __init__(self, shape, interpolation=Image.BILINEAR):
        assert isinstance(shape, tuple)
        self.shape = (shape[1], shape[0])
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        ratio = self.shape[0] / self.shape[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t-w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t-h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.shape, self.interpolation)

        return np.array(img)


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor


class DeNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

        return tensor


class ToTensor(object):
    def __init__(self, d255=False):
        self.d255 = d255

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # print(img.ndim)
        if img.ndim == 3:
            img = np.transpose(img, [2, 0, 1])
        if self.d255:
            img = img / 255.
        img = torch.from_numpy(img)
        return img.float()


class ToLabel(object):
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        img = torch.from_numpy(img)
        return img.long()


class ReLabel(object):
    def __init__(self, old_label, new_label):
        self.old_label = old_label
        self.new_label = new_label

    def __call__(self, img):
        # assert isinstance(img, torch.LongTensor)
        img[img == self.old_label] = self.new_label
        return img


def get_image_transform_list(size, normalize):
    transform_list = [
        ReSize(size, interpolation=Image.BILINEAR),
        ToTensor(d255=True),
        Normalize(**normalize)
    ]
    return transform_list


def get_mask_transform_list(size):
    transform_list = [
        ReSize(size, interpolation=Image.NEAREST),
        ToLabel(),
        ReLabel(255, 1)
    ]
    return transform_list


def get_heatmap_transform_list(size):
    transform_list = [
        ReSize(size, interpolation=Image.CUBIC),
        ToTensor()
    ]
    return transform_list
