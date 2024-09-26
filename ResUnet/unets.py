# -*- coding: utf-8 -*-
"""
3D U-Net architectures

Tim Scherr
Last Update: 05.09.2019
"""

import torch
import torch.nn as nn
from torchsummary import summary

import layer as layer


def build_model(configs, device, print_summary=False):
    """Set up a model with selected architecture.

    :param configs: Dictionary containing data paths and information for the training process.
        :type configs: dict
    :param device: cuda (gpu) or cpu.
        :type device:
    :param print_summary: Boolean to print a model summary or not.
        :type print_summary: bool
    :return: model
    """

    ch_out = 1

    # Build model
    if configs['architecture'] == '2Dunet':

        model = UNet2D(ch_in=1, ch_out=ch_out, filters=configs['filters'], order=configs['order'])
    
    elif configs['architecture'] == 'unet':

        model = UNet3D(ch_in=1, ch_out=ch_out, filters=configs['filters'], order=configs['order'])

    elif configs['architecture'] == 'resunet':

        model = ResUNet3D(ch_in=1, ch_out=ch_out, filters=configs['filters'], order=configs['order'])

    elif configs['architecture'] == 'tnet':

        model = TNet(ch_in=1, ch_out=ch_out, filters=configs['filters'], order=configs['order'])

    elif configs['architecture'] == 'unet_2branches':

        model = UNet3D_2Branches(ch_in=1, ch_out=ch_out, filters=configs['filters'], order=configs['order'])

    elif configs['architecture'] == 'DUNet':
        
        model = DUNet(ch_in=1, pool_method='conv', act_fun='relu', normalization='bn', filters=(64, 1024))
    
    else:

        raise Exception('Architecture "{}" is not available'.format(configs['architecture']))

    # Use multiple GPUs if available
    if configs['num_gpus'] > 1:
        model = nn.DataParallel(model)

    # Move model to used device (GPU or CPU)
    model = model.to(device)

    # Print model parameters
    if print_summary:
        summary(model, (1, configs['shape'][0], configs['shape'][1], configs['shape'][2]), configs['batch_size'])

    return model


class TNet(nn.Module):
    """3D U-net derivation with specialised spatial filters for every dimension.

    """

    def __init__(self, ch_in=1, ch_out=1, filters=[64, 128, 256, 512], order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Channels of the input.
            :type ch_in: int
        :param ch_out: Channels of the output.
            :type ch_out: int
        :param num_conv: Number of convolutions in each block.
            :type num_conv: int
        :param filters: Number of feature maps used in every encoder block.
            :type filters: list
        """

        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.order = order

        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.encoder.append(layer.SpatialConvBlock3D(ch_in=self.ch_in, ch_out=self.filters[0], order=self.order,
                                                        num_conv=self.num_conv))
        self.pool.append(nn.Conv3d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=2, stride=2,
                                    padding=0, bias=True))
        for i_depth in range(1, len(self.filters)):
            self.encoder.append(layer.SpatialConvBlock3D(ch_in=self.filters[i_depth], ch_out=self.filters[i_depth],
                                                            order=self.order, num_conv=self.num_conv))
        for i_depth in range(1, len(self.filters)-1):
            self.pool.append(nn.Conv3d(in_channels=self.filters[i_depth], out_channels=self.filters[i_depth+1],
                                        kernel_size=2, stride=2, padding=0, bias=True))

        # Decoder: transposed convolutions
        self.decoder_transp = nn.ModuleList()
        for i_depth in range(len(self.filters) - 1):
            self.decoder_transp.append(layer.TranspConvBlock3D(ch_in=self.filters[-i_depth - 1],
                                                                ch_out=self.filters[-i_depth - 2], order=self.order))

        # Decoder: convolutions
        self.decoder_conv = nn.ModuleList()
        for i_depth in range(len(self.filters) - 1):
            self.decoder_conv.append(
                layer.ConvBlock3D(ch_in=self.filters[-i_depth - 1], ch_out=self.filters[-i_depth - 2],
                                    order=self.order, kernel_size=self.kernel_size, num_conv=self.num_conv))

        # Last 1x1 convolution
        self.decoder_conv.append(nn.Conv3d(in_channels=self.filters[0], out_channels=self.ch_out, kernel_size=1,
                                            stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()
        # Encoder
        for i in range(len(self.encoder) - 1):
            x = self.encoder[i](x)
            x_temp.append(x)
            x = self.pool[i](x)
        x = self.encoder[-1](x)

        # Decoder
        x_temp = list(reversed(x_temp))
        for i in range(len(self.filters) - 1):
            x = self.decoder_transp[i](x)
            x = torch.cat([x, x_temp[i]], 1)
            x = self.decoder_conv[i](x)
        x = self.decoder_conv[-1](x)

        return x


class UNet2D(nn.Module):
    """Implementation of the 3D U-Net architecture.

    Changes to original architecture: Feature maps within an encoder/decoder block are not doubled.

    Reference: Cicek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation". In:
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer. 2016.

    """

    def __init__(self, ch_in=1, ch_out=1, filters=[64, 128, 256, 512, 1024], order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Channels of the input.
            :type ch_in: int
        :param ch_out: Channels of the output.
            :type ch_out: int
        :param num_conv: Number of convolutions in each block.
            :type num_conv: int
        :param filters: Number of feature maps used in every encoder block.
            :type filters: list
        """

        super().__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.order = order
            
        # Maximum pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(layer.ConvBlock2D(ch_in=self.ch_in, ch_out=self.filters[0], order=self.order,
                                              kernel_size=self.kernel_size, num_conv=self.num_conv))
        for i_depth in range(1, len(self.filters)):
            self.encoder.append(layer.ConvBlock2D(ch_in=self.filters[i_depth-1], ch_out=self.filters[i_depth],
                                                  order=self.order, kernel_size=self.kernel_size,
                                                  num_conv=self.num_conv))
            
        # Decoder: transposed convolutions
        self.decoder_transp = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_transp.append(layer.TranspConvBlock2D(ch_in=self.filters[-i_depth-1],
                                                               ch_out=self.filters[-i_depth-2], order=self.order))
        
        # Decoder: convolutions
        self.decoder_conv = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_conv.append(layer.ConvBlock2D(ch_in=self.filters[-i_depth-1], ch_out=self.filters[-i_depth-2],
                                                       order=self.order, kernel_size=self.kernel_size,
                                                       num_conv=self.num_conv))
        
        # Last 1x1 convolution
        self.decoder_conv.append(nn.Conv2d(in_channels=self.filters[0], out_channels=self.ch_out, kernel_size=1,
                                           stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()
        # Encoder
        for i in range(len(self.encoder)-1):
            x = self.encoder[i](x)
            x_temp.append(x)
            x = self.maxpool(x)
        x = self.encoder[-1](x)

        # Decoder
        x_temp = list(reversed(x_temp))
        for i in range(len(self.filters)-1):
            x = self.decoder_transp[i](x)
            x = torch.cat([x, x_temp[i]], 1)
            x = self.decoder_conv[i](x)
        x = self.decoder_conv[-1](x)
        
        return x


class UNet3D(nn.Module):
    """Implementation of the 3D U-Net architecture.

    Changes to original architecture: Feature maps within an encoder/decoder block are not doubled.

    Reference: Cicek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation". In:
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer. 2016.

    """

    def __init__(self, ch_in=1, ch_out=1, filters=[64, 128, 256, 512, 1024], order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Channels of the input.
            :type ch_in: int
        :param ch_out: Channels of the output.
            :type ch_out: int
        :param num_conv: Number of convolutions in each block.
            :type num_conv: int
        :param filters: Number of feature maps used in every encoder block.
            :type filters: list
        """

        super().__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.order = order
            
        # Maximum pooling
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(layer.ConvBlock3D(ch_in=self.ch_in, ch_out=self.filters[0], order=self.order,
                                              kernel_size=self.kernel_size, num_conv=self.num_conv))
        for i_depth in range(1, len(self.filters)):
            self.encoder.append(layer.ConvBlock3D(ch_in=self.filters[i_depth-1], ch_out=self.filters[i_depth],
                                                  order=self.order, kernel_size=self.kernel_size,
                                                  num_conv=self.num_conv))
            
        # Decoder: transposed convolutions
        self.decoder_transp = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_transp.append(layer.TranspConvBlock3D(ch_in=self.filters[-i_depth-1],
                                                               ch_out=self.filters[-i_depth-2], order=self.order))
        
        # Decoder: convolutions
        self.decoder_conv = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_conv.append(layer.ConvBlock3D(ch_in=self.filters[-i_depth-1], ch_out=self.filters[-i_depth-2],
                                                       order=self.order, kernel_size=self.kernel_size,
                                                       num_conv=self.num_conv))
        
        # Last 1x1 convolution
        self.decoder_conv.append(nn.Conv3d(in_channels=self.filters[0], out_channels=self.ch_out, kernel_size=1,
                                           stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()
        # Encoder
        for i in range(len(self.encoder)-1):
            x = self.encoder[i](x)
            x_temp.append(x)
            x = self.maxpool(x)
        x = self.encoder[-1](x)

        # Decoder
        x_temp = list(reversed(x_temp))
        for i in range(len(self.filters)-1):
            x = self.decoder_transp[i](x)
            x = torch.cat([x, x_temp[i]], 1)
            x = self.decoder_conv[i](x)
        x = self.decoder_conv[-1](x)
        
        return x


class ResUNet3D(nn.Module):
    """ 3D U-Net with residual connections
    """

    def __init__(self, ch_in=1, ch_out=1, filters=[64, 128, 256, 512, 1024], order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Channels of the input.
            :type ch_in: int
        :param ch_out: Channels of the output.
            :type ch_out: int
        :param num_conv: Number of convolutions in each block.
            :type num_conv: int
        :param filters: Number of feature maps used in every encoder block.
            :type filters: list
        """

        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.order = order

        # Maximum pooling
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(layer.ResConvBlock3D(ch_in=self.ch_in, ch_out=self.filters[0], order=self.order,
                                                 kernel_size=self.kernel_size))
        for i_depth in range(1, len(self.filters)):
            self.encoder.append(layer.ResConvBlock3D(ch_in=self.filters[i_depth - 1], ch_out=self.filters[i_depth],
                                                     order=self.order, kernel_size=self.kernel_size))

        # Decoder: transposed convolutions
        self.decoder_transp = nn.ModuleList()
        for i_depth in range(len(self.filters) - 1):
            self.decoder_transp.append(layer.TranspConvBlock3D(ch_in=self.filters[-i_depth-1],
                                                               ch_out=self.filters[-i_depth-2], order=self.order))

        # Decoder: convolutions
        self.decoder_conv = nn.ModuleList()
        for i_depth in range(len(self.filters) - 1):
            self.decoder_conv.append(
                layer.ConvBlock3D(ch_in=self.filters[-i_depth - 1], ch_out=self.filters[-i_depth - 2],
                                  order=self.order, kernel_size=self.kernel_size, num_conv=self.num_conv))

        # Last 1x1 convolution
        self.decoder_conv.append(nn.Conv3d(in_channels=self.filters[0], out_channels=self.ch_out, kernel_size=1,
                                           stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()
        # Encoder
        for i in range(len(self.encoder) - 1):
            x = self.encoder[i](x)
            x_temp.append(x)
            x = self.maxpool(x)
        x = self.encoder[-1](x)

        # Decoder
        x_temp = list(reversed(x_temp))
        for i in range(len(self.filters) - 1):
            x = self.decoder_transp[i](x)
            x = torch.cat([x, x_temp[i]], 1)
            x = self.decoder_conv[i](x)
        x = self.decoder_conv[-1](x)

        return x



class UNet3D_2Branches(nn.Module):
    """Implementation of the 3D U-Net architecture.

    Changes to original architecture: Feature maps within an encoder/decoder block are not doubled.

    Reference: Cicek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation". In:
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer. 2016.

    """

    def __init__(self, ch_in=1, ch_out=1, filters=[64, 128, 256, 512, 1024], order='rb', kernel_size=3, num_conv=2):
        """

        :param ch_in: Channels of the input.
            :type ch_in: int
        :param ch_out: Channels of the output.
            :type ch_out: int
        :param num_conv: Number of convolutions in each block.
            :type num_conv: int
        :param filters: Number of feature maps used in every encoder block.
            :type filters: list
        """

        super().__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_conv = num_conv
        self.order = order
            
        # Maximum pooling
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(layer.ConvBlock3D(ch_in=self.ch_in, ch_out=self.filters[0], order=self.order,
                                              kernel_size=self.kernel_size, num_conv=self.num_conv))
        for i_depth in range(1, len(self.filters)):
            self.encoder.append(layer.ConvBlock3D(ch_in=self.filters[i_depth-1], ch_out=self.filters[i_depth],
                                                  order=self.order, kernel_size=self.kernel_size,
                                                  num_conv=self.num_conv))
            

        # Decoder: transposed convolutions
        self.decoder_transp_1 = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_transp_1.append(layer.TranspConvBlock3D(ch_in=self.filters[-i_depth-1],
                                                            ch_out=self.filters[-i_depth-2], order=self.order))
        
        # Decoder: convolutions
        self.decoder_conv_1 = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_conv_1.append(layer.ConvBlock3D(ch_in=self.filters[-i_depth-1], ch_out=self.filters[-i_depth-2],
                                                    order=self.order, kernel_size=self.kernel_size,
                                                    num_conv=self.num_conv))
        
        # Last 1x1 convolution
        self.decoder_conv_1.append(nn.Conv3d(in_channels=self.filters[0], out_channels=1, kernel_size=1,
                                        stride=1, padding=0))

        



        # Decoder: transposed convolutions
        self.decoder_transp_2 = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_transp_2.append(layer.TranspConvBlock3D(ch_in=self.filters[-i_depth-1],
                                                            ch_out=self.filters[-i_depth-2], order=self.order))
        
        # Decoder: convolutions
        self.decoder_conv_2 = nn.ModuleList()
        for i_depth in range(len(self.filters)-1):
            self.decoder_conv_2.append(layer.ConvBlock3D(ch_in=self.filters[-i_depth-1], ch_out=self.filters[-i_depth-2],
                                                    order=self.order, kernel_size=self.kernel_size,
                                                    num_conv=self.num_conv))
        
        # Last 1x1 convolution
        self.decoder_conv_2.append(nn.Conv3d(in_channels=self.filters[0], out_channels=1, kernel_size=1,
                                        stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()
        # Encoder
        for i in range(len(self.encoder)-1):
            x = self.encoder[i](x)
            x_temp.append(x)
            x = self.maxpool(x)
        x = self.encoder[-1](x)

        # Decoder
        x_temp = list(reversed(x_temp))
        x_1 = x
        x_2 = x
        for i in range(len(self.filters)-1):
            x_1 = self.decoder_transp_1[i](x_1)
            x_1 = torch.cat([x_1, x_temp[i]], 1)
            x_1 = self.decoder_conv_1[i](x_1)
        x_1 = self.decoder_conv_1[-1](x_1)

        for i in range(len(self.filters)-1):
            x_2 = self.decoder_transp_2[i](x_2)
            x_2 = torch.cat([x_2, x_temp[i]], 1)
            x_2 = self.decoder_conv_2[i](x_2)
        x_2 = self.decoder_conv_2[-1](x_2)
        
        x = torch.cat([x_1,x_2], 1)
        
        return x


class DUNet(nn.Module):
    """ Distance transform U-net. """

    def __init__(self, ch_in=1, pool_method='conv', act_fun='relu', normalization='bn', filters=(64, 1024)):
        """

        :param ch_in:
        :param pool_method:
        :param act_fun:
        :param normalization:
        :param filters:
        """

        super().__init__()

        self.ch_in = ch_in
        self.filters = filters
        self.pool_method = pool_method

        # Encoder
        self.encoderConv = nn.ModuleList()

        if self.pool_method == 'max':
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
        elif self.pool_method == 'conv':
            self.pooling = nn.ModuleList()

        # First encoder block
        n_featuremaps = filters[0]
        self.encoderConv.append(ConvBlock(ch_in=self.ch_in,
                                          ch_out=n_featuremaps,
                                          act_fun=act_fun,
                                          normalization=normalization))
        if self.pool_method == 'conv':
            self.pooling.append(ConvPool(ch_in=n_featuremaps, act_fun=act_fun, normalization=normalization))

        # Remaining encoder blocks
        while n_featuremaps < filters[1]:

            self.encoderConv.append(ConvBlock(ch_in=n_featuremaps,
                                              ch_out=(n_featuremaps*2),
                                              act_fun=act_fun,
                                              normalization=normalization))

            if n_featuremaps * 2 < filters[1] and self.pool_method == 'conv':
                self.pooling.append(ConvPool(ch_in=n_featuremaps*2, act_fun=act_fun, normalization=normalization))

            n_featuremaps *= 2

        # Decoder 1 (borders, seeds) and Decoder 2 (cells)
        self.decoder1Upconv = nn.ModuleList()
        self.decoder1Conv = nn.ModuleList()
        self.decoder2Upconv = nn.ModuleList()
        self.decoder2Conv = nn.ModuleList()

        while n_featuremaps > filters[0]:
            self.decoder1Upconv.append(TranspConvBlock(ch_in=n_featuremaps,
                                                       ch_out=(n_featuremaps // 2),
                                                       normalization=normalization))
            self.decoder1Conv.append(ConvBlock(ch_in=n_featuremaps,
                                               ch_out=(n_featuremaps // 2),
                                               act_fun=act_fun,
                                               normalization=normalization))
            self.decoder2Upconv.append(TranspConvBlock(ch_in=n_featuremaps,
                                                       ch_out=(n_featuremaps // 2),
                                                       normalization=normalization))
            self.decoder2Conv.append(ConvBlock(ch_in=n_featuremaps,
                                               ch_out=(n_featuremaps // 2),
                                               act_fun=act_fun,
                                               normalization=normalization))
            n_featuremaps //= 2

        # Last 1x1 convolutions
        self.decoder1Conv.append(nn.Conv3d(n_featuremaps, 1, kernel_size=1, stride=1, padding=0))
        self.decoder2Conv.append(nn.Conv3d(n_featuremaps, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """

        :param x: Model input.
            :type x:
        :return: Model output / prediction.
        """

        x_temp = list()

        # Encoder
        for i in range(len(self.encoderConv) - 1):
            x = self.encoderConv[i](x)
            x_temp.append(x)
            if self.pool_method == 'max':
                x = self.pooling(x)
            elif self.pool_method == 'conv':
                x = self.pooling[i](x)
        x = self.encoderConv[-1](x)

        # Intermediate results for concatenation
        x_temp = list(reversed(x_temp))

        # Decoder 1 (borders + seeds)
        for i in range(len(self.decoder1Conv) - 1):
            if i == 0:
                x1 = self.decoder1Upconv[i](x)
            else:
                x1 = self.decoder1Upconv[i](x1)
            x1 = torch.cat([x1, x_temp[i]], 1)
            x1 = self.decoder1Conv[i](x1)
        x1 = self.decoder1Conv[-1](x1)

        # Decoder 2 (cells)
        for i in range(len(self.decoder2Conv) - 1):
            if i == 0:
                x2 = self.decoder2Upconv[i](x)
            else:
                x2 = self.decoder2Upconv[i](x2)
            x2 = torch.cat([x2, x_temp[i]], 1)
            x2 = self.decoder2Conv[i](x2)
        x2 = self.decoder2Conv[-1](x2)

        return torch.cat([x1,x2], 1)


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (nn.Tanh(nn.Softplus(x)))
        return x


class ConvBlock(nn.Module):
    """ Basic convolutional block of a U-net. """

    def __init__(self, ch_in, ch_out, act_fun, normalization):
        """

        :param ch_in:
        :param ch_out:
        :param act_fun:
        :param normalization:
        """

        super().__init__()
        self.conv = list()

        # 1st convolution
        self.conv.append(nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 1st activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 1st normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm3d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm3d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        # 2nd convolution
        self.conv.append(nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        # 2nd activation function
        if act_fun == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        # 2nd normalization
        if normalization == 'bn':
            self.conv.append(nn.BatchNorm3d(ch_out))
        elif normalization == 'gn':
            self.conv.append(nn.GroupNorm(num_groups=8, num_channels=ch_out))
        elif normalization == 'in':
            self.conv.append(nn.InstanceNorm3d(num_features=ch_out))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv)):
            x = self.conv[i](x)

        return x


class ConvPool(nn.Module):

    def __init__(self, ch_in, act_fun, normalization):
        """

        :param ch_in:
        :param act_fun:
        :param normalization:
        """

        super().__init__()
        self.conv_pool = list()

        self.conv_pool.append(nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=2, padding=1, bias=True))

        if act_fun == 'relu':
            self.conv_pool.append(nn.ReLU(inplace=True))
        elif act_fun == 'leakyrelu':
            self.conv_pool.append(nn.LeakyReLU(inplace=True))
        elif act_fun == 'elu':
            self.conv_pool.append(nn.ELU(inplace=True))
        elif act_fun == 'mish':
            self.conv_pool.append(Mish())
        else:
            raise Exception('Unsupported activation function: {}'.format(act_fun))

        if normalization == 'bn':
            self.conv_pool.append(nn.BatchNorm3d(ch_in))
        elif normalization == 'gn':
            self.conv_pool.append(nn.GroupNorm(num_groups=8, num_channels=ch_in))
        elif normalization == 'in':
            self.conv_pool.append(nn.InstanceNorm3d(num_features=ch_in))
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

        self.conv_pool = nn.Sequential(*self.conv_pool)

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (feature maps).
        """
        for i in range(len(self.conv_pool)):
            x = self.conv_pool[i](x)

        return x


class TranspConvBlock(nn.Module):
    """ Upsampling block of a unet (with transposed convolutions). """

    def __init__(self, ch_in, ch_out, normalization):
        """

        :param ch_in:
        :param ch_out:
        :param normalization:
        """
        super().__init__()

        self.up = nn.Sequential(nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2))
        if normalization == 'bn':
            self.norm = nn.BatchNorm3d(ch_out)
        elif normalization == 'gn':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=ch_out)
        elif normalization == 'in':
            self.norm = nn.InstanceNorm3d(num_features=ch_out)
        else:
            raise Exception('Unsupported normalization: {}'.format(normalization))

    def forward(self, x):
        """

        :param x: Block input (image or feature maps).
            :type x:
        :return: Block output (upsampled feature maps).
        """
        x = self.up(x)
        x = self.norm(x)

        return x