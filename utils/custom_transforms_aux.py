import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask_aux = sample['label_aux']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        mask_aux = np.array(mask_aux).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        mask /= 255.0
        mask = np.round(mask)
        return {'image': img,
                'label': mask,
                'label_aux':mask_aux}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        mask_aux = sample['label_aux']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        mask_aux = np.array(mask_aux).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        mask_aux = torch.from_numpy(mask_aux).float()

        return {'image': img,
                'label': mask,
                'label_aux':mask_aux}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask_aux = sample['label_aux']


        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        mask_aux = mask_aux.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'label_aux':mask_aux}

class Normalize_img(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        img = np.array(image).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img

class Normalize_mask(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __call__(self, mask):
        mask = np.array(mask).astype(np.float32)
        mask /= 255.0
        mask = np.round(mask)
        return mask

class ToTensor_img(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

class ToTensor_mask(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, mask):
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()
        return mask

class FixedResize_img(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img):
        img = img.resize(self.size, Image.BILINEAR)

        return img

class FixedResize_mask(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, mask):
        mask = mask.resize(self.size, Image.NEAREST)

        return mask