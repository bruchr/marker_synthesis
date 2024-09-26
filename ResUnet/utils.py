# -*- coding: utf-8 -*-
"""
Useful stuff.

Tim Scherr
Last Update: 05.09.2019
"""

import os

import numpy as np
from scipy import ndimage
import torch


def mkdirs_results(configs):
    """Make directories for the dataloader check and for the trained models.

    :param configs: Dictionary containing the result path.
        :type configs: dict
    :return: None
    """

    if not os.path.exists(configs['path_results'] + '/models'):
        os.mkdir(configs['path_results'] + '/models')

    # Predictions
    if not os.path.exists(configs['path_results'] + '/' + configs['run_name']):
        os.mkdir(configs['path_results'] + '/' + configs['run_name'])

    return None

def find_values(img):
    """ Find appearing intensity values or cell nuclei in intensity-coded label image.

    :param img: Intensity-coded label image.
        :type:
    :return: List of appearing intensity values / cell nuclei in the image.
    """

    hist = np.histogram(img, bins=range(1, img.max() + 2), range=(1, img.max() + 1))

    # Exclude values that are not present in the label image
    values = np.delete(hist[1], np.where(hist[0] == 0))
    values = values[:-1]

    return values


def min_max_normalization(img, min_value=None, max_value=None, noise_level=None):
    """ Min-max-normalization for (volumetric) uint images.

    :param img: Image with shape [depth, height, width], [height, width] or [height, width, color channel].
        :type img:
    :param min_value: Minimum value for the normalization. All values below this value are clipped
        :type min_value: int
    :param max_value: Maximum value for the normalization. All values above this value are clipped.
        :type max_value: int
    :param noise_level: If (max_value - min_value) < noise_level, the image, which probably only contains noise, is
        normalized to the range of the used data type (appears dark instead of bright).
    :return: Normalized (volumetric) image (float32)
    """

    if max_value is None:  # Get new maximum value for the normalization. Filter the image to avoid hot/cold pixels

        img_filtered = ndimage.filters.median_filter(input=img, size=(1, 3, 3))

        max_value = img_filtered.max()

        if min_value is None:  # Get new minimum value

            min_value = img_filtered.min()

    if noise_level is not None:  # Avoid the normalization of images, that only contain noise

        if (max_value - min_value) < noise_level:
            max_value = np.iinfo(img.dtype).max
            min_value = 0

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply Min-max-normalization
    # img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1 # Range [-1,1]
    img = (img.astype(np.float32) - min_value) / (max_value - min_value) # Range [0,1]

    return img.astype(np.float32)


def to_one_hot(y, num_classes=None):
    """Convert int tensor with n dimensions to one-hot representation with num_classes classes (there is already a
    PyTorch built-in to_one_hot function accepted for a future release).

    :param y: Tensor
        :type y:
    :param num_classes: Number of classes / channels of the one-hot representation.
        :type num_classes: int
    :return: One-hot-encoded tensor (channels last).
    """
    
    y_tensor = y.clone().detach()
    y_tensor = y_tensor.type(torch.LongTensor).to(y.device).view(-1, 1)
    n_dims = num_classes if num_classes is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).to(y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    return y_one_hot.to(y.device)


def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values