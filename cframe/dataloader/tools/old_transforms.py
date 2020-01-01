import math
import numbers
import random
from PIL import Image, ImageOps


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, label):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            label = ImageOps.expand(label, border=self.padding, fill=0)

        assert img.size == label.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, label
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), label.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), label.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), label.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __init__(self, ratio=0.5):
        self.ratio = 0.5

    def __call__(self, img, label):
        if random.random() < self.ratio:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label


class FreeScale(object):
    def __init__(self, size):
        size = (size, size)
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, label):
        assert img.size == label.size
        return img.resize(self.size, Image.BILINEAR), label.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, label
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), label.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), label.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), label.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, label))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, label):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), label.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, label):
        assert img.size == label.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, label = img.resize((w, h), Image.BILINEAR), label.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, label))


JOINT_TRANS_DICT = {
    'random_crop': RandomCrop,
    'random_size': RandomSized,
    'random_size_crop': RandomSizedCrop,
    'random_rotate': RandomRotate,
    'random_hflip': RandomHorizontallyFlip,
    'center_crop': CenterCrop,
    'scale': Scale,
    'free_scale': FreeScale
}


class JointTransformCompose(object):
    def  __init__(self, config, phase):
        self.config = config
        self.phase = phase
        self.config_joint_trans = self.config[phase]['joint']
        self.joint_transforms = dict()

        for trans_key in self.config_joint_trans['seq']:
            self.joint_transforms[trans_key] = JOINT_TRANS_DICT[trans_key](
                **self.config_joint_trans[trans_key]
            )

    def __call__(self, image, label):
        for trans_key in self.config_joint_trans['seq']:
            image, label = self.joint_transforms[trans_key](image, label)
        return image, label

