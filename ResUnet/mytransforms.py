# -*- coding: utf-8 -*-
"""
Transformations/augmentations for 3D cell nuclei segmentation

Tim Scherr
Last update: 05.09.2019
"""
import cv2
import numpy as np
from imgaug import augmenters as iaa
import random
import scipy
from skimage.util import random_noise
import torch
import torchvision.transforms as transforms

#np.seterr(all='raise')

def augmentors(augmentation, p):
    """Get the augmentations/transformations for the training and validation set.

    :param augmentation: String to select the wanted augmentation.
        :type augmentation: str
    :param p: Probability to apply the selected augmentation.
        :type p: float
    :param label_type: Type of the given label image, e.g., boundary label image.
        :type label_type: bool
    :return Dictionary containing the PyTorch transformations for the training and validation set.
    """

    if augmentation == 'none':

        data_transforms = {'train': ToTensor(),
                           'val': ToTensor()}

    elif augmentation == 'noise':

        data_transforms = {'train': transforms.Compose([RandomNoise(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'flip':

        data_transforms = {'train': transforms.Compose([RandomFlip(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'rotate':

        data_transforms = {'train': transforms.Compose([RandomRotate(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'perspectiveAffine':

        data_transforms = {'train': transforms.Compose([Perspective(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'emboss':

        data_transforms = {'train': transforms.Compose([Emboss(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'blur':

        data_transforms = {'train': transforms.Compose([RandomBlur(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'scale':

        data_transforms = {'train': transforms.Compose([Scaling(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'contrast':

        data_transforms = {'train': transforms.Compose([Contrast(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'elaTrans':

        data_transforms = {'train': transforms.Compose([ElaTrans(p=p), ToTensor()]),
                           'val': ToTensor()}

    elif augmentation == 'combo_low':

        data_transforms = {'train': transforms.Compose([RandomFlip(p=0.5),
                                                        RandomRotate(p=0.1),
                                                        Scaling(p=0.1),
                                                        Contrast(p=0.1),
                                                        Emboss(p=0.1),
                                                        Perspective(p=0.1),
                                                        RandomNoise(p=0.1),
                                                        RandomBlur(p=0.1),
                                                        ToTensor()]),
                           'val': ToTensor()
                           }

    elif augmentation == 'combo_med':
        data_transforms = {'train': transforms.Compose([RandomFlip(p=0.5),
                                                        RandomRotate(p=0.2),
                                                        Scaling(p=0.2),
                                                        Contrast(p=0.2),
                                                        Emboss(p=0.2),
                                                        Perspective(p=0.8),
                                                        RandomNoise(p=0.2),
                                                        RandomBlur(p=0.2),
                                                        ToTensor()]),
                           'val': ToTensor()
                           }

    elif augmentation == 'combo_high':
        data_transforms = {'train': transforms.Compose([RandomFlip(p=0.5),
                                                        RandomRotate(p=0.3),
                                                        Scaling(p=0.3),
                                                        Contrast(p=0.3),
                                                        Emboss(p=0.2),
                                                        Perspective(p=0.3),
                                                        RandomNoise(p=0.3),
                                                        RandomBlur(p=0.3),
                                                        ToTensor()]),
                           'val': ToTensor()
                           }

    elif augmentation == 'combo_own':
        data_transforms = {'train': transforms.Compose([RandomFlip(p=0.5),
                                                        RandomRotate(p=0.3),
                                                        Scaling(p=0.3),
                                                        Contrast(p=0.3),
                                                        Emboss(p=0.2),
                                                        ElaTrans(p=0.8),
                                                        RandomNoise(p=0.3),
                                                        RandomBlur(p=0.2),
                                                        ToTensor()]),
                           'val': ToTensor()
                           }

    elif augmentation == 'combo_distance':
        data_transforms = {'train': transforms.Compose([RandomFlip(p=0.5),
                                                        RandomRotate(p=0.3),
                                                        Scaling(p=0.3),
                                                        Contrast(p=0.3),
                                                        RandomNoise(p=0.3),
                                                        RandomBlur(p=0.2),
                                                        ToTensor()]),
                           'val': ToTensor()
                           }

    else:
        raise Exception('Unknown transformation: {}'.format(augmentation))

    return data_transforms


class Contrast(object):
    """Contrast augmentation (label-preserving transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        contrast_range = (0.65, 1.35)
        gamma_range = (0.5, 1.5)
        epsilon = 1e-7,

        if random.random() < self.p:# and img.max() >= -0.5:

            # Contrast
            img_mean, img_min, img_max = img.mean(), img.min(), img.max()
            factor = np.random.uniform(contrast_range[0], contrast_range[1])
            img = (img - img_mean) * factor + img_mean
            # assert img.min() > -1, "(Contrast - Contrast): Min value of image is smaller than -1 : min:{}".format(img.min())

            # Gamma
            img_mean, img_std, img_min, img_max = img.mean(), img.std(), img.min(), img.max()
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            rnge = img_max - img_min
            img = np.power(((img - img_min) / float(rnge + epsilon)), gamma) * rnge + img_min
            # assert img.min() > -1, "(Contrast - Gamma): Min value of image is smaller than -1 : min:{}".format(img.min())

            if random.random() < 0.5:# and (img.max()-img.min()) > 0.1:
                img = img - img.mean() + img_mean
                img = img / (img.std() + 1e-8) * img_std
                # assert img.min() > -1, "(Contrast - Gamma-Mean): Min value of image is smaller than -1 : min:{}".format(img.min())

            img = np.clip(img, -1, 1)

        # assert np.any(label[:,0]!=0), '(Contrast): Only ignored label in image'
        # assert not np.isnan(img).any(), '(Contrast): NaN value in image'
        # assert not np.isnan(label).any(), '(Contrast): NaN value in label'
        # assert img.min() < img.max(), "(Contrast): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        rnge = img.max()-img.min()
        if rnge!=0:
            return {'image': img.copy(), 'label': label.copy()}
        else:
            return {'image': sample['image'].copy(), 'label': sample['label'].copy()}


class Emboss(object):
    """Emboss image and overlay the result with the original image (label-changing transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:

            alpha = (0.2, 0.7)
            strength = (0.3, 0.8)

            seq = iaa.Sequential([iaa.Emboss(alpha=alpha, strength=strength, deterministic=False)])
            matrix = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 1]])
            seq2 = iaa.Sequential([iaa.convolutional.Convolve(matrix=matrix)])

            if len(img.shape)==4:
                for i in range(img.shape[1]):

                    # Add Emboss to selected images
                    img[0, i, :, :] = seq.augment_image(img[0, i, :, :])

                    # Shift label image
                    label[0, i, :, :] = seq2.augment_image(label[0, i, :, :])

            else:
                 # Add Emboss to selected images
                img[0, :, :] = seq.augment_image(img[0, :, :])

                # Shift label image
                label[0, :, :] = seq2.augment_image(label[0, :, :])

            img = np.clip(img, -1, 1)

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class Perspective(object):
    """Apply a perspective transformation (label-changing transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if len(img.shape)==4:
            c, d, h, w = img.shape
        else:
            c, h, w = img.shape

        if random.random() < self.p:

            pts1 = np.float32([[random.randint(0, h // 5), random.randint(0, w // 5)],
                               [random.randint(4 * h // 5, h), random.randint(0, w // 5)],
                               [random.randint(0, h // 5), random.randint(4 * w // 5, w)],
                               [random.randint(4 * h // 5, h), random.randint(4 * w // 5, w)]])
            pts2 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])

            if len(img.shape)==4:
                for i in range(img.shape[1]):

                    perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)
                    img[0, i, :, :] = np.expand_dims(cv2.warpPerspective(img[0, i, :, :], perspective_transform, (h, w)),
                                                    axis=-1)[:, :, 0]
                    label[0, i, :, :] = np.expand_dims(cv2.warpPerspective(label[0, i, :, :], perspective_transform, (h, w),
                                                                        flags=cv2.INTER_NEAREST),
                                                    axis=-1).astype(np.uint8)[:, :, 0]
            else:
                perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)
                img[0, :, :] = np.expand_dims(cv2.warpPerspective(img[0, :, :], perspective_transform, (h, w)),
                                                    axis=-1)[:, :, 0]
                label[0, :, :] = np.expand_dims(cv2.warpPerspective(label[0, :, :], perspective_transform, (h, w),
                                                                    flags=cv2.INTER_NEAREST),
                                                axis=-1).astype(np.uint8)[:, :, 0]

            img = np.clip(img, -1, 1)

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class RandomBlur(object):
    """Blur images using a gaussian kernels or median filter (label-preserving transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        # Unpack dictionary
        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:
            h = random.randint(0, 1)

            # Gaussian blur
            if h == 0:
                sigma = 2.5 * random.random()
                img = scipy.ndimage.gaussian_filter(img, sigma, order=0)

            # Median filter
            else:
                if len(img.shape)==4:
                    img = scipy.ndimage.median_filter(img, size=(1, 3, 3, 3))
                else:
                    img = scipy.ndimage.median_filter(img, size=(1, 3, 3))

        # assert np.any(label[:,0]!=0), '(RandomBlur): Only ignored label in image'
        # assert not np.isnan(img).any(), '(RandomBlur): NaN value in image'
        # assert not np.isnan(label).any(), '(RandomBlur): NaN value in label'
        # assert img.min() < img.max(), "(RandomBlur): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class RandomFlip(object):
    """Flip or rotate (90°) image and label image (label-changing transformation)."""

    def __init__(self, p=0.5):
        """
        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """
         :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """
        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:

            if len(img.shape)==4:
                # img.shape: (Channels, Depth, Height, Width)
                h = random.randint(0, 3)
                if h == 0:
                    # Flip left-right
                    axis = 3 if label.ndim==4 else 4
                    img = np.flip(img, axis=3)
                    label = np.flip(label, axis=axis)
                elif h == 1:
                    # Flip up-down
                    axis = 2 if label.ndim==4 else 3
                    img = np.flip(img, axis=2)
                    label = np.flip(label, axis=axis)
                elif h == 2:
                    # Rotate 90°
                    axis = (2, 3) if label.ndim==4 else (3, 4)
                    img = np.rot90(img, axes=(2, 3))
                    label = np.rot90(label, axes=axis)
                elif h == 3:
                    # Flip in depth dimension
                    axis = 1 if label.ndim==4 else 2
                    img = np.flip(img, axis=1)
                    label = np.flip(label, axis=axis)
            else:
                # img.shape: (Channels, Depth, Height, Width)
                h = random.randint(0, 2)
                if h == 0:
                    # Flip left-right
                    img = np.flip(img, axis=2)
                    label = np.flip(label, axis=2)
                elif h == 1:
                    # Flip up-down
                    img = np.flip(img, axis=1)
                    label = np.flip(label, axis=1)
                elif h == 2:
                    # Rotate 90°
                    img = np.rot90(img, axes=(1, 2))
                    label = np.rot90(label, axes=(1, 2))
                # elif h == 3:
                #     # Flip in depth dimension
                #     img = np.flip(img, axis=1)
                #     label = np.flip(label, axis=1)

        # assert np.any(label[:,0]!=0), '(RandomFlip): Only ignored label in image'
        # assert not np.isnan(img).any(), '(RandomFlip): NaN value in image'
        # assert not np.isnan(label).any(), '(RandomFlip): NaN value in label'
        # assert img.min() < img.max(), "(RandomFlip): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img.copy(), 'label': label.copy()}


class RandomNoise(object):
    """Add additive Gaussian noise (label-preserving transformation)."""

    def __init__(self, p=0.25):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        if random.random() < self.p:

            h = random.randint(0, 2)
            if h == 0:
                # Add Poisson noise
                # assert np.isfinite(img).all(), "Some image elements are not finite: max:{}; min:{}".format(img.max(),img.min())
                # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
                himg = (img - img.min()) / (img.max() - img.min()) * 65535
                noise = 20 * (himg - np.random.poisson(himg.astype(np.uint16))) / 65535
                img = np.clip(img + noise.astype(np.float32), -1, 1)

            elif h == 1:
                # Add Gaussian noise with sigma 5-10% of image range
                sigma = random.randint(5, 10) / 100

                img = img + np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
                img = np.clip(img, -1, 1)

            # Add salt and pepper noise:
            if h == 2:
                img = random_noise(img, mode='s&p')

        # assert np.any(label[:,0]!=0), '(RandomNoise): Only ignored label in image'
        # assert not np.isnan(img).any(), '(RandomNoise): NaN value in image'
        # assert not np.isnan(label).any(), '(RandomNoise): NaN value in label'
        # assert img.min() < img.max(), "(RandomNoise): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class RandomRotate(object):
    """Rotate randomly (label-changing transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary cointaining augmented image and label image (numpy arrays).
        """

        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        
        angle = (-180, 180)

        if random.random() < self.p:

            angle = random.uniform(angle[0], angle[1])

            if len(img.shape)==4:
                for i in range(img.shape[1]):

                    seq1 = iaa.Sequential([iaa.Affine(rotate=angle, deterministic=True)])
                    seq2 = iaa.Sequential([iaa.Affine(rotate=angle, deterministic=True, order=0)])
                    img[0, i, :, :] = np.expand_dims(seq1.augment_image(img[0, i, :, :] + 1), axis=0) - 1
                    if label.ndim == 4:
                        label[0, i, :, :] = np.expand_dims(seq2.augment_image(label[0, i, :, :].astype(np.uint8)), axis=0)
                    elif label.ndim == 5:
                        for i_channel in range(label.shape[1]):
                            label[0, i_channel, i, :, :] = np.expand_dims(seq2.augment_image(label[0, i_channel, i, :, :]), axis=0)
                    else:
                        raise Exception('Unsupported number of label dimensions: {}'.format(label.ndim))
            else:
                seq1 = iaa.Sequential([iaa.Affine(rotate=angle, deterministic=True)])
                seq2 = iaa.Sequential([iaa.Affine(rotate=angle, deterministic=True, order=0)])
                img[0, :, :] = np.expand_dims(seq1.augment_image(img[0, :, :] + 1), axis=0) - 1
                label[0, :, :] = np.expand_dims(seq2.augment_image(label[0, :, :].astype(np.uint8)), axis=0)

        # assert np.any(label[:,0]!=0), '(RandomRotate): Only ignored label in image'
        # assert not np.isnan(img).any(), '(RandomRotate): NaN value in image'
        # assert not np.isnan(label).any(), '(RandomRotate): NaN value in label'
        # assert img.min() < img.max(), "(RandomRotate): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class Scaling(object):
    """Scaling of an image: cropping or zero-padding is used to keep the image shape constant (label-changing
    transformation)."""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        img, label = sample['image'], sample['label']

        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        scale_xy_1 = random.uniform(0.8, 1.3)
        scale_xy_2 = random.uniform(0.8, 1.3)
        if len(img.shape)==4:
            scale_yz_1 = random.uniform(0.9, 1.2)
            scale_yz_2 = random.uniform(0.9, 1.2)
            scale_xz_1 = random.uniform(0.9, 1.2)
            scale_xz_2 = random.uniform(0.9, 1.2)

        if random.random() < self.p:

            if len(img.shape)==4:
                # In xy-dim
                for i in range(img.shape[1]):
                    seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale_xy_1, "y": scale_xy_2})])
                    seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale_xy_1, "y": scale_xy_2}, order=0)])
                    img[0, i, :, :] = seq1.augment_image(img[0, i, :, :] + 1) - 1
                    if label.ndim == 4:
                        label[0, i, :, :] = seq2.augment_image(label[0, i, :, :]).astype(np.uint8)
                    elif label.ndim == 5:
                        for i_channel in range(label.shape[1]):
                            label[0, i_channel, i, :, :] = seq2.augment_image(label[0, i_channel, i, :, :])
                    else:
                        raise Exception('Unsupported number of label dimensions: {}'.format(label.ndim))

                # In yz-dim
                if random.random() < 0.4:
                    for i in range(img.shape[3]):
                        seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale_yz_1, "y": scale_yz_2})])
                        seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale_yz_1, "y": scale_yz_2}, order=0)])
                        img[0, :, :, i] = seq1.augment_image(img[0, :, :, i] + 1) - 1
                        if label.ndim == 4:
                            label[0, :, :, i] = seq2.augment_image(label[0, :, :, i]).astype(np.uint8)
                        elif label.ndim == 5:
                            for i_channel in range(label.shape[1]):
                                label[0, i_channel, :, :, i] = seq2.augment_image(label[0, i_channel, :, :, i])
                        else:
                            raise Exception('Unsupported number of label dimensions: {}'.format(label.ndim))

                # In xz-dim
                if random.random() < 0.4:
                    for i in range(img.shape[2]):
                        seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale_xz_1, "y": scale_xz_2})])
                        seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale_xz_1, "y": scale_xz_2}, order=0)])
                        img[0, :, i, :] = seq1.augment_image(img[0, :, i, :] + 1) - 1
                        if label.ndim == 4:
                            label[0, :, i, :] = seq2.augment_image(label[0, :, i, :]).astype(np.uint8)
                        elif label.ndim == 5:
                            for i_channel in range(label.shape[1]):
                                label[0, i_channel, :, i, :] = seq2.augment_image(label[0, i_channel, :, i, :])
                        else:
                            raise Exception('Unsupported number of label dimensions: {}'.format(label.ndim))
            else:
                # In xy-dim
                seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale_xy_1, "y": scale_xy_2})])
                seq2 = iaa.Sequential([iaa.Affine(scale={"x": scale_xy_1, "y": scale_xy_2}, order=0)])
                img[0, :, :] = seq1.augment_image(img[0, :, :] + 1) - 1
                label[0, :, :] = seq2.augment_image(label[0, :, :]).astype(np.uint8)

            img = np.clip(img, -1, 1)

        # assert np.any(label[:,0]!=0), '(Scaling): Only ignored label in image'
        # assert not np.isnan(img).any(), '(Scaling): NaN value in image'
        # assert not np.isnan(label).any(), '(Scaling): NaN value in label'
        # assert img.min() < img.max(), "(Scaling): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        return {'image': img, 'label': label}


class ElaTrans(object):
    """Elastic transform of an image: cropping or zero-padding is used to keep the image shape constant (label-changing
    transformation).
    Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.
    https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Dictionary containing augmented image and label image (numpy arrays).
        """

        # def draw_grid(im, grid_size):
        #     # Draw grid lines
        #     for k in range(0, im.shape[0], grid_size[0]):
        #         for i in range(0, im.shape[2], grid_size[2]):
        #             cv2.line(im[k,:,:], (i, 0), (i, im.shape[1]), color=(1,))
        #         for j in range(0, im.shape[1], grid_size[1]):
        #             cv2.line(im[k,:,:], (0, j), (im.shape[2], j), color=(1,))


        img, label = sample['image'], sample['label']
        img, label = img[0,:,:,:], label[0,:,:,:]
        
        alpha = [50,200,200]
        sigma = [4,7.5,7.5]

        # draw_grid(img, (10,20,20))
        # draw_grid(label, (10,20,20))

        from scipy.ndimage.interpolation import map_coordinates
        from scipy.ndimage.filters import gaussian_filter
        # import tifffile as tiff
        random_state = np.random.RandomState(None)

        shape = img.shape
        dz = scipy.ndimage.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[0], mode="constant", cval=0) * alpha[0]
        dy = scipy.ndimage.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[1], mode="constant", cval=0) * alpha[1]
        dx = scipy.ndimage.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[2], mode="constant", cval=0) * alpha[2]

        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),indexing='ij')
        indices = np.reshape(z+dz, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # tiff.imsave('D:/Bruch/transformation/' + 'img.tif', img)
        # tiff.imsave('D:/Bruch/transformation/' + 'label.tif', label)

        distorted_image = scipy.ndimage.interpolation.map_coordinates(img, indices, order=1, mode='reflect')
        img = distorted_image.reshape(img.shape)
        distorted_label = scipy.ndimage.interpolation.map_coordinates(label, indices, order=1, mode='reflect')
        label = distorted_label.reshape(img.shape)

        # tiff.imsave('D:/Bruch/transformation/' + 'ET_img.tif', img)
        # tiff.imsave('D:/Bruch/transformation/' + 'ET_label.tif', label)

        img, label = np.expand_dims(img,axis=0), np.expand_dims(label,axis=0)
        return {'image': img, 'label': label}


class ToTensor(object):
    """Convert image and label image to Torch tensors."""

    def __init__(self):
        pass
        
    def __call__(self, sample):
        """

        :param sample: Dictionary containing image and label image (numpy arrays).
            :type sample: dict
        :return: Normalized image and label image as Torch tensors
        """

        img, label = sample['image'], sample['label']
        
        # assert img.min() < img.max(), "Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())

        img = img.astype(np.float32)

        img = torch.from_numpy(img).to(torch.float)

        # binary crossentropy loss needs float tensor [batch size, channels, depth, height, width] as target
        label = torch.from_numpy(label).to(torch.float)

        # assert np.any(label[:,0]!=0), '(ToTensor): Only ignored label in image'
        assert not torch.isnan(img).any(), '(ToTensor): NaN value in image'
        # assert not np.isnan(label).any(), '(ToTensor): NaN value in label'
        # assert img.min() < img.max(), "(ToTensor): Min value of image is smaller or equal to max value : max:{}; min:{}".format(img.max(),img.min())
        
        return img, label
