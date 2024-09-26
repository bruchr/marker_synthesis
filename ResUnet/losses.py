# -*- coding: utf-8 -*-
"""
Loss functions (for 3D segmentation)

Tim Scherr
Last Update: 05.09.2019
"""

import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils import to_one_hot


def get_loss(loss_function):
    """Get optimization criterion / loss function.

    :param loss_function: String to call the wanted optimization criterion / loss function.
        :type loss_function: str
    :return: Optimization criterion / loss function.
    """

    if loss_function == 'bce_dice':

        criterion = bce_dice

    elif loss_function == 'ce_dice':

        criterion = ce_dice

    elif loss_function == 'bce':

        criterion = nn.BCEWithLogitsLoss()

    elif loss_function == 'ce':

        criterion = nn.CrossEntropyLoss()

    elif loss_function == 'ce_ignore':

        criterion = ce_ignore

    elif loss_function == 'wce_ignore':

        criterion = wce_ignore

    elif loss_function == 'dice':

        criterion = dice_loss

    elif loss_function == 'wbce_dice':

        criterion = wbce_dice

    elif loss_function == 'wce_dice':

        criterion = wce_dice

    elif loss_function == 'bce_ignore':

        criterion = bce_ignore
    
    elif loss_function == 'wbce_ignore':

        criterion = wbce_ignore

    elif loss_function == 'wbce_ignore_2D':

        criterion = wbce_ignore_2d
    
    elif loss_function == 'wbce_dice_ignore':
        
        criterion = wbce_dice_ignore

    elif loss_function == 'smooth_l1':

        criterion = smooth_l1
    
    elif loss_function == 'mse':

        criterion = torch.nn.MSELoss()

    elif loss_function == 'wmse':

        criterion = wmse
        
    else:

        raise Exception('Loss function "{}" not known!'.format(loss_function))

    return criterion


def dice_loss(y_pred, y_true, use_sigmoid=True):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, depth, height, width].
        :type y_true:
    :param use_sigmoid: Apply sigmoid activation function to the prediction y_pred.
        :type use_sigmoid: bool
    :return:
    """

    smooth = 1.

    gt = y_true.contiguous().view(-1)
    
    if use_sigmoid:
        pred = torch.sigmoid(y_pred)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred.contiguous().view(-1)
        
    pred_gt = torch.sum(gt * pred)
    
    loss = 1 - (2. * pred_gt + smooth) / (torch.sum(gt**2) + torch.sum(pred**2) + smooth)
    
    return loss


def bce_dice(y_pred, y_true):
    """ Sum of binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, depth, height, width].
        :type y_true:
    :return:
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)
    
    return loss


def wbce_dice(y_pred, y_true):
    """ Sum of weighted binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, depth, height, width].
        :type y_true:
    :return:
    """

    eps = 1e-9

    w0 = 1 / torch.sqrt(torch.sum(y_true) + eps) * y_true
    w1 = 1 / torch.sqrt(torch.sum(1 - y_true)) * (1 - y_true)

    weight_map = w0 + w1
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true))
    loss = bce_loss + 1.5 * dice_loss(y_pred, y_true)

    return loss


def ce_dice(y_pred, y_true, num_classes=3):
    """Sum of crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    # Crossentropy Loss
    loss_ce = nn.CrossEntropyLoss() 
    ce_loss = loss_ce(y_pred, y_true)

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :, :], y_true_one_hot[:, index, :, :, :],
                                        use_sigmoid=False)
    
    return ce_loss + 0.5 * dice_score


def ce_ignore(y_pred, y_true, num_classes=4):
    """crossentropy loss with ignored class

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()

    y_true[y_true==0] = 1
    y_true = y_true-1

    eps = 1e-9

    weight_map = 1-y_true_one_hot[:, 0:1, :, :, :]

    loss_ce = nn.CrossEntropyLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_ce(y_pred, y_true))

    return ce_loss


def wce_ignore(y_pred, y_true, num_classes=4):
    """crossentropy loss with ignored class

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()

    y_true[y_true==0] = 1
    y_true = y_true-1

    eps = 1e-9

    # Weighted CrossEntropy Loss
    #w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :, :]) + eps) * y_true_one_hot[:, 0, :, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :, :]) + eps) * y_true_one_hot[:, 1, :, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :, :]) + eps) * y_true_one_hot[:, 2, :, :, :]
    w3 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 3, :, :, :]) + eps) * y_true_one_hot[:, 3, :, :, :]

    weight_map = w1 + w2 + w3
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :, :]

    #weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_ce = nn.CrossEntropyLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_ce(y_pred, y_true))

    return ce_loss


def wce_dice(y_pred, y_true, num_classes=3):
    """Sum of weighted crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    eps = 1e-9

    # Weighted CrossEntropy Loss
    w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :, :]) + eps) * y_true_one_hot[:, 0, :, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :, :]) + eps) * y_true_one_hot[:, 1, :, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :, :]) + eps) * y_true_one_hot[:, 2, :, :, :]

    weight_map = w0 + w1 + w2
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :, :]

    weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_ce = nn.CrossEntropyLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_ce(y_pred, y_true))

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :, :], y_true_one_hot[:, index, :, :, :],
                                        use_sigmoid=False)

    return 0.5 * ce_loss + 0.3 * dice_score


def bce_ignore(y_pred, y_true, num_classes=3):
    """Sum of weighted crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()

    eps = 1e-9

    # Weighted CrossEntropy Loss
    weight_map = 1-y_true_one_hot[:, 0:1, :, :, :]

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true_one_hot[:,2:3,:,:,:]))

    return ce_loss


def wbce_ignore(y_pred, y_true, num_classes=3):
    """Sum of weighted crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()

    eps = 1e-9

    # Weighted CrossEntropy Loss
    #w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :, :]) + eps) * y_true_one_hot[:, 0, :, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :, :]) + eps) * y_true_one_hot[:, 1, :, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :, :]) + eps) * y_true_one_hot[:, 2, :, :, :]

    weight_map = w1 + w2
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :, :]

    #weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true_one_hot[:,2:3,:,:,:]))

    return ce_loss


def wbce_ignore_2d(y_pred, y_true, num_classes=3):
    """Sum of weighted crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, depth, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 3, 1, 2]).float()

    eps = 1e-9

    # Weighted CrossEntropy Loss
    #w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :, :]) + eps) * y_true_one_hot[:, 0, :, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :]) + eps) * y_true_one_hot[:, 1, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :]) + eps) * y_true_one_hot[:, 2, :, :]

    weight_map = w1 + w2
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :]

    #weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    ce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true_one_hot[:,2:3,:,:]))

    return ce_loss


def wbce_dice_ignore(y_pred, y_true, num_classes=3):
    """ Sum of weighted binary crossentropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, depth, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, depth, height, width].
        :type y_true:
    :return:
    """

    y_true_one_hot = to_one_hot(y_true, num_classes).permute([0, 4, 1, 2, 3]).float()

    eps = 1e-9

    # Weighted CrossEntropy Loss
    #w0 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 0, :, :, :]) + eps) * y_true_one_hot[:, 0, :, :, :]
    w1 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 1, :, :, :]) + eps) * y_true_one_hot[:, 1, :, :, :]
    w2 = 1 / torch.sqrt(torch.sum(y_true_one_hot[:, 2, :, :, :]) + eps) * y_true_one_hot[:, 2, :, :, :]

    weight_map = w1 + w2
    weight_map = torch.sum(weight_map.view(-1)) * weight_map

    weight_map = weight_map[:, None, :, :, :]

    #weight_map = gaussian_smoothing_3d(weight_map, 1, 5, 0.9)

    loss_bce = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = torch.mean(weight_map * loss_bce(y_pred, y_true_one_hot[:,2:3,:,:,:]))

    
    smooth = 1.

    gt = y_true_one_hot[:,2:3,:,:,:].contiguous().view(-1)
    y_pred_valid = y_pred * (1-y_true_one_hot[:, 0:1, :, :, :])
    if True:#use_sigmoid:
        pred = torch.sigmoid(y_pred_valid)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred_valid.contiguous().view(-1)
        
    pred_gt = torch.sum(gt * pred)
    
    dice_loss_own = 1 - (2. * pred_gt + smooth) / (torch.sum(gt**2) + torch.sum(pred**2) + smooth)



    loss = bce_loss + 1.5 * dice_loss_own

    return loss


def gaussian_smoothing_3d(x, channels, kernel_size, sigma):
    """Smoothing to enlarge/weight weight maps.

    :param x: Input tensor [batch size, channels, depth, height, width]
        :type x:
    :param channels: Number of image channels.
        :type channels: int
    :param kernel_size: Size of the 3d kernel for the convolution.
        :type kernel_size: int
    :param sigma: Standard deviation of the kernel.
        :type sigma: float
    :return: Convolved tensor.
    """

    kernel_size = [kernel_size] * 3
    sigma = [sigma] * 3

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=2, bias=False)
    conv = conv.to('cuda')
    conv.weight.data = kernel.to('cuda')
    conv.weight.requires_grad = False

    return conv(x)


def smooth_l1(y_pred, y_true, num_classes=3):
    loss_l1 = nn.SmoothL1Loss(reduction='none')

    loss = torch.mean(y_true[:,0,:,:,:] * (loss_l1(y_pred[:,0,:,:,:],y_true[:,1,:,:,:]) + loss_l1(y_pred[:,1,:,:,:],y_true[:,2,:,:,:])) )

    # assert not torch.isnan(loss).any(), 'NaN in loss calculation'

    return loss


def wmse(y_pred, y_true):
    mse_loss = nn.MSELoss(reduction='none')
    weight_map = y_true/torch.mean(y_true)
    wmse_loss = torch.mean(weight_map * mse_loss(y_pred, y_true))

    return wmse_loss
