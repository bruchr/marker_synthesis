# -*- coding: utf-8 -*-
"""
Basic CNN building blocks for the segmentation of volumetric data.

Tim Scherr
Last Update: 05.09.2019
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import tanh


class Mish(nn.Module):
    """ Mish activatisn function. """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (tanh(F.softplus(x)))
        return x


class ConvBlock3D(nn.Module):
    """ Basic convolutional block. """

    def __init__(self, ch_in, ch_out, order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the activation function and normalization to apply.
                'e', 'l', 'm', 'r': ELU, leaky ReLu, mish, ReLU activation function
                'b', 'g', 'i': batch, group, instance normalization
                'rb': ReLU + batch normalization
                'gl': group normalization + leaky ReLU
                ...
            :type order: str
        :param kernel_size: Size of the kernel used for the convolutions.
            :type kernel_size:
        :param num_conv: Number of convolutions.
            :type num_conv: int
        """

        super().__init__()

        self.conv = []

        for i_reps in range(num_conv):

            # Convolution
            self.conv.append(nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=1,
                                       padding=1, bias=True))

            # Activation function and normalization
            for char in order:

                if char == 'b':  # Batch normalization

                    self.conv.append(nn.BatchNorm3d(num_features=ch_out))

                elif char == 'e':  # ELU

                    self.conv.append(nn.ELU(inplace=True))

                elif char == 'g':  # Group normalization

                    self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

                elif char == 'i':  # Instance normalization

                    self.conv.append(nn.InstanceNorm3d(num_features=ch_out))

                elif char == 'l':  # Leaky ReLU activation function

                    self.conv.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

                elif char == 'm':  # Mish activation function

                    self.conv.append(Mish())

                elif char == 'r':  # ReLU activation function

                    self.conv.append(nn.ReLU(inplace=True))

                else:

                    raise Exception("Unsupported layer type.")

            # After first repetition the Conv3d input has more channels
            ch_in = ch_out

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """

        :param x: Module input.
            :type x:
        :return: Module output.
        """
        x = self.conv(x)

        return x


class ConvBlock2D(nn.Module):
    """ Basic convolutional block. """

    def __init__(self, ch_in, ch_out, order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the activation function and normalization to apply.
                'e', 'l', 'm', 'r': ELU, leaky ReLu, mish, ReLU activation function
                'b', 'g', 'i': batch, group, instance normalization
                'rb': ReLU + batch normalization
                'gl': group normalization + leaky ReLU
                ...
            :type order: str
        :param kernel_size: Size of the kernel used for the convolutions.
            :type kernel_size:
        :param num_conv: Number of convolutions.
            :type num_conv: int
        """

        super().__init__()

        self.conv = []

        for i_reps in range(num_conv):

            # Convolution
            self.conv.append(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=1,
                                       padding=1, bias=True))

            # Activation function and normalization
            for char in order:

                if char == 'b':  # Batch normalization

                    self.conv.append(nn.BatchNorm2d(num_features=ch_out))

                elif char == 'e':  # ELU

                    self.conv.append(nn.ELU(inplace=True))

                elif char == 'g':  # Group normalization

                    self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

                elif char == 'i':  # Instance normalization

                    self.conv.append(nn.InstanceNorm2d(num_features=ch_out))

                elif char == 'l':  # Leaky ReLU activation function

                    self.conv.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

                elif char == 'm':  # Mish activation function

                    self.conv.append(Mish())

                elif char == 'r':  # ReLU activation function

                    self.conv.append(nn.ReLU(inplace=True))

                else:

                    raise Exception("Unsupported layer type.")

            # After first repetition the Conv3d input has more channels
            ch_in = ch_out

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """

        :param x: Module input.
            :type x:
        :return: Module output.
        """
        x = self.conv(x)

        return x


class SpatialConvBlock3D(nn.Module):
    """ Basic convolutional block specialised spatial filters for every dimension. """

    def __init__(self, ch_in, ch_out, order='rb', num_conv=2):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the activation function and normalization to apply.
                'e', 'l', 'm', 'r': ELU, leaky ReLu, mish, ReLU activation function
                'b', 'g', 'i': batch, group, instance normalization
                'rb': ReLU + batch normalization
                'gl': group normalization + leaky ReLU
                ...
            :type order: str
        :param num_conv: Number of convolutions.
            :type num_conv: int
        """

        super().__init__()

        self.conv = []
        self.conv_xy = []
        self.conv_yz = []
        self.conv_xz = []

        for i_reps in range(num_conv):

            # Convolution
            self.conv.append(nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 3), stride=1,
                                       padding=1, bias=True))

            self.conv_xy.append(nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(1, 3, 3), stride=1,
                                          padding=(0, 1, 1), bias=True))

            self.conv_xz.append(nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 1, 3), stride=1,
                                          padding=(1, 0, 1), bias=True))

            self.conv_yz.append(nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3, 1), stride=1,
                                          padding=(1, 1, 0), bias=True))

            # Activation function and normalization
            for char in order:

                if char == 'b':  # Batch normalization

                    self.conv.append(nn.BatchNorm3d(num_features=ch_out))
                    self.conv_xy.append(nn.BatchNorm3d(num_features=ch_out))
                    self.conv_yz.append(nn.BatchNorm3d(num_features=ch_out))
                    self.conv_xz.append(nn.BatchNorm3d(num_features=ch_out))

                elif char == 'e':  # ELU activation function

                    self.conv.append(nn.ELU(inplace=True))
                    self.conv_xy.append(nn.ELU(inplace=True))
                    self.conv_xz.append(nn.ELU(inplace=True))
                    self.conv_yz.append(nn.ELU(inplace=True))

                elif char == 'g':  # Group normalization

                    self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
                    self.conv_xy.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
                    self.conv_yz.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
                    self.conv_xz.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

                elif char == 'i':  # Instance normalization

                    self.conv.append(nn.InstanceNorm3d(num_features=ch_out))
                    self.conv_xy.append(nn.InstanceNorm3d(num_features=ch_out))
                    self.conv_yz.append(nn.InstanceNorm3d(num_features=ch_out))
                    self.conv_xz.append(nn.InstanceNorm3d(num_features=ch_out))

                elif char == 'l':  # Leaky ReLU activation function

                    self.conv.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                    self.conv_xy.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                    self.conv_xz.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                    self.conv_yz.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

                elif char == 'm':  # Mish activation function

                    self.conv.append(Mish())
                    self.conv_xy.append(Mish())
                    self.conv_xz.append(Mish())
                    self.conv_yz.append(Mish())

                elif char == 'r':  # ReLU activation function

                    self.conv.append(nn.ReLU(inplace=True))
                    self.conv_xy.append(nn.ReLU(inplace=True))
                    self.conv_xz.append(nn.ReLU(inplace=True))
                    self.conv_yz.append(nn.ReLU(inplace=True))

                else:

                    raise Exception("Unsupported layer type.")

            # After first repetition the Conv3d input has more channels
            ch_in = ch_out

        self.conv = nn.Sequential(*self.conv)
        self.conv_xy = nn.Sequential(*self.conv_xy)
        self.conv_yz = nn.Sequential(*self.conv_yz)
        self.conv_xz = nn.Sequential(*self.conv_xz)

    def forward(self, x):
        """

        :param x: Module input.
            :type x:
        :return: Module output.
        """
        xy = self.conv_xy(x)
        xz = self.conv_xz(x)
        yz = self.conv_yz(x)
        x = self.conv(x)

        x = x + xy + xz + yz

        return x


class ResConvBlock3D(nn.Module):
    """Convolutional block consisting of a single convolution (needed to make sure that the number of input and output
     channels in the residual block match and the addition works) and a residual block.
    """

    def __init__(self, ch_in, ch_out, order='lg', kernel_size=3):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the activation function and normalization to apply.
                'e', 'l', 'm', 'r': ELU, leaky ReLu, mish, ReLU activation function
                'b', 'g', 'i': batch, group, instance normalization
                'rb': ReLU + batch normalization
                'gl': group normalization + leaky ReLU
                ...
            :type order: str
        :param kernel_size: Size of the kernel used for the convolutions.
            :type kernel_size:
        """

        super().__init__()

        self.res = []
        self.res_add = []

        # Single convolution
        self.conv = ConvBlock3D(ch_in=ch_in, ch_out=ch_out, order=order, kernel_size=kernel_size, num_conv=1)

        # Residual block
        self.res.append(nn.Conv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=kernel_size, stride=1, padding=1,
                                  bias=True))

        for char in order:

            if char == 'b':  # Batch normalization

                self.res.append(nn.BatchNorm3d(num_features=ch_out))
                self.res_add.append(nn.BatchNorm3d(num_features=ch_out))

            elif char == 'e':  # ELU activation function

                self.res.append(nn.ELU(inplace=True))
                self.res_add.append(nn.ELU(inplace=True))

            elif char == 'g':  # Group normalization

                self.res.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
                self.res_add.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

            elif char == 'i':  # Instance normalization

                self.res.append(nn.InstanceNorm3d(num_features=ch_out))
                self.res_add.append(nn.InstanceNorm3d(num_features=ch_out))

            elif char == 'l':  # Leaky ReLU activation function

                self.res.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                self.res_add.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

            elif char == 'm':  # Mish activation function

                self.res.append(Mish())
                self.res_add.append(Mish())

            elif char == 'r':  # ReLU activation function

                self.res.append(nn.ReLU(inplace=True))
                self.res_add.append(nn.ReLU(inplace=True))

            else:

                raise Exception("Unsupported layer type.")

        self.res.append(nn.Conv3d(in_channels=ch_out, out_channels=ch_out, kernel_size=kernel_size, stride=1, padding=1,
                                  bias=True))
        self.res = nn.Sequential(*self.res)
        self.res_add = nn.Sequential(*self.res_add)

    def forward(self, x):
        """

        :param x: Module input (image or feature maps).
            :type x:
        :return: Module output (feature maps).
        """
        x = self.conv(x)
        x_res = x
        x = self.res(x)
        x = self.res_add(x + x_res)

        return x


class TranspConvBlock3D(nn.Module):
    """ Basic transposed convolutional block. """

    def __init__(self, ch_in, ch_out, order='b'):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the normalization to apply.
                'b', 'g', 'i': batch, group, instance normalization
            :type order: str
        """
        super().__init__()

        self.transp_conv = []

        # Transposed convolution
        self.transp_conv.append(nn.ConvTranspose3d(in_channels=ch_in, out_channels=ch_out, kernel_size=2, stride=2))

        # Normalization
        for char in order:

            if char == 'b':  # Batch normalization

                self.transp_conv.append(nn.BatchNorm3d(num_features=ch_out))

            elif char == 'g':  # Group normalization

                self.transp_conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

            elif char == 'i':  # Instance normalization

                self.transp_conv.append(nn.InstanceNorm3d(num_features=ch_out))

        self.transp_conv = nn.Sequential(*self.transp_conv)

    def forward(self, x):
        """

        :param x: Module input (image or feature maps).
            :type x:
        :return: Module output (upsampled feature maps).
        """
        x = self.transp_conv(x)

        return x


class TranspConvBlock2D(nn.Module):
    """ Basic transposed convolutional block. """

    def __init__(self, ch_in, ch_out, order='b'):
        """

        :param ch_in: Number of input channels / feature maps of the module.
            :type ch_in: int
        :param ch_out: Number of output channels / feature maps of the module.
            :type ch_out: int
        :param order: Order and selection of the normalization to apply.
                'b', 'g', 'i': batch, group, instance normalization
            :type order: str
        """
        super().__init__()

        self.transp_conv = []

        # Transposed convolution
        self.transp_conv.append(nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=2, stride=2))

        # Normalization
        for char in order:

            if char == 'b':  # Batch normalization

                self.transp_conv.append(nn.BatchNorm2d(num_features=ch_out))

            elif char == 'g':  # Group normalization

                self.transp_conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))

            elif char == 'i':  # Instance normalization

                self.transp_conv.append(nn.InstanceNorm2d(num_features=ch_out))

        self.transp_conv = nn.Sequential(*self.transp_conv)

    def forward(self, x):
        """

        :param x: Module input (image or feature maps).
            :type x:
        :return: Module output (upsampled feature maps).
        """
        x = self.transp_conv(x)

        return x
