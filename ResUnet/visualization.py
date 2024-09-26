import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer():
    """Visualize the loss of CycleGAN"""

    def __init__(self, config):
        logdir = os.path.join("runs", config["name"])
        self.writer = SummaryWriter(logdir)
        self.loss_groups = {}

    def write_losses(self, loss, epoch):
        for group, loss_names in self.loss_groups.items():
            loss_group = {}
            for key, loss_name in enumerate(loss_names):
                loss_group[loss_name] = loss[loss_name]
                self.writer.add_scalar(group+"/"+loss_name, loss[loss_name], epoch)
            #self.writer.add_scalars(group, loss_group, epoch)
            #self.writer.close()
        self.writer.close()

    def write_images(self, images, epoch):
        for key, image in images.items():
            image_np = tensor2np(torch.squeeze(image[0,:]), "uint8")
            if image_np.ndim == 2:
                self.writer.add_image(key, image_np, epoch, dataformats="HW")
            elif image_np.ndim == 3:
                mid_ind = int(np.floor(image_np.shape[0]/2))
                self.writer.add_image(key, image_np[mid_ind,...], epoch, dataformats="HW")
            else:
                self.writer.add_image(key, image_np, epoch)
        self.writer.close()

    def define_loss_group(self, group_name, group_items):
        self.loss_groups[group_name] = group_items


def tensor2np(input_image, imtype):
    """ Converts a Tensor image to a numpy image

    Arguments:
        input_image {tensor} -- input image tensor
        imtype {string} -- Data type of the numpy image (e.g. uint8, uint16)

    Returns:
        [np.array] -- image as numpy array
    """
    image_tensor = input_image.data
    image_numpy = image_tensor.cpu().float().numpy()
    # (image+1)/2 because tanh (GAN otuput) is [-1,1] therefore +1 => [0,2] and /2 => [0,1] then * max of image value stretches to imagesize
    image_numpy = image_numpy * np.iinfo(imtype).max  # 255.0 np.iinfo(imtype).max  
    # image_numpy = (image_numpy + 1) / 2.0 * np.iinfo(imtype).max  # 255.0 np.iinfo(imtype).max  
    return image_numpy.astype(imtype)
