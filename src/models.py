import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import torch.optim as optim
import os
import copy
from torchsummary import summary
import queue
import copy
from PIL import Image
from torchvision import transforms

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = (3, 3), padding = (1, 1), dilation = (1, 1), stride = (1, 1), final_layer = False, bias = True):
        super(DoubleConv2D, self).__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
                # nn.BatchNorm2d(out_channels),
                # nn.Linear(),
            )
            conv_layers.append(double_conv1)
        else:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            conv_layers.append(double_conv1)
        self.double_conv = nn.Sequential(*conv_layers)
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,  kernel_size = 3, padding = 2, dilation = 1, stride = 1, final_layer = False):
        super().__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular'),
            # nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                # nn.BatchNorm1d(out_channels),
                # nn.Sigmoid(),
            )
            conv_layers.append(double_conv1)
        else:
            double_conv1 = nn.Sequential(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
            conv_layers.append(double_conv1)
        self.double_conv = nn.Sequential(*conv_layers)
    def forward(self, x):
        return self.double_conv(x)

class Up1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super(Up1D, self).__init__()
        self.conv = DoubleConv1D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)

    def forward(self, x1):
        x1 = F.interpolate(x1, scale_factor= 2)
        x = x1
        return self.conv(x)

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 1, dilation = 1, bias = True):
        super(Up2D, self).__init__()
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation, bias = bias)

    def forward(self, x1):
        x = F.interpolate(x1, scale_factor= 2)
        return self.conv(x)

class Down1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.conv = DoubleConv1D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)
        
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class Down2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 1, dilation = 1, stride = 1, bias = True):
        super(Down2D, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation, stride =1, bias = bias)
        
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu')):
        super().__init__()
        self.down_layers = []
        self.up_layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.inlayer = DoubleConv1D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, stride = 2)
        # self.midlayer = DoubleConv1D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.outlayer = DoubleConv1D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, final_layer = True)

        # for d in range(depth):
        #     downlayer = Down1D(mid_channels * 2 ** d, mid_channels * (2** (d + 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
        #     self.down_layers.append(downlayer.to(device))
        
        # for d in range(depth):
        #     # print(d)
        #     uplayer = Up1D(mid_channels * (2 ** (depth - d)), mid_channels * (2**(depth - d - 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
        #     self.up_layers.append(uplayer.to(device))
        for d in range(depth):
            downlayer = Down1D(mid_channels, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.down_layers.append(downlayer.to(device))
        
        for d in range(depth):
            # print(d)
            uplayer = Up1D(mid_channels, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.up_layers.append(uplayer.to(device))

    def forward(self, x):
        
        # x = torch.view(x.size(0), self.in_channels, -1)
        x = self.inlayer(x)
        # print(x.size())
        down_queue = queue.LifoQueue(maxsize= len(self.down_layers))
        for down in self.down_layers:
            down_queue.put(x)
            x = down(x)
            # print('down: ', x.size())
        # x = self.midlayer(x)
        for up in self.up_layers:
            x_queued = down_queue.get()
            x = up(x)
            x = x + x_queued
        x = self.outlayer(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu') , sig_layer = True, bias = True):
        
        super(UNet2D, self).__init__()
        self.inlayer = DoubleConv2D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.midlayer = DoubleConv2D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.down1 = Down2D(mid_channels, mid_channels * 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.down2 = Down2D(mid_channels * 2, mid_channels * 4, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.down3 = Down2D(mid_channels * 4, mid_channels * 8, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.down4 = Down2D(mid_channels * 8, mid_channels * 16, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.down5 = Down2D(mid_channels * 16, mid_channels * 32, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.down6 = Down2D(mid_channels * 32, mid_channels * 64, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)

        # self.up6 = Up2D(mid_channels* 64, mid_channels * 32, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.up5 = Up2D(mid_channels* 32, mid_channels * 16, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.up4 = Up2D(mid_channels* 16, mid_channels * 8, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.up3 = Up2D(mid_channels* 8, mid_channels * 4, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.up2 = Up2D(mid_channels* 4, mid_channels * 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up1 = Up2D(mid_channels* 2, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # print(sig_layer)

        self.outlayer = DoubleConv2D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, final_layer= sig_layer, bias = bias)
        # self.drop = nn.Dropout(p = 0.5, inplace = False)
    def forward(self, x):
        # print(x.size())
        x = self.inlayer(x)
        x = self.down1(x)
        # x2 = self.down2(x1)
        # x3 = self.down3(x2)
        # x4 = self.down4(x3)
        # x5 = self.down5(x4)
        # x6 = self.down6(x5)
        # x = self.up6(x6)
        # # # print('up: ', x.size())
        # x = self.up5(x )
        # # # print('up: ', x.size())
        # x = self.up4(x + x4)
        # # # print('up: ', x.size())
        # x = self.up3(x )
        # # # print('up: ', x.size())
        # x = self.up2(x + x2)
        # # x = self.drop(x)
        # print('up: ', x.size())
        x = self.up1(x)

        return self.outlayer(x)

class DenseDown1D(nn.Module):
    def __init__(self, signal_dim = 128, down_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim * in_filters
        self.down_sample_factor = down_sample_factor
        width = int(signal_dim * n_filters / self.down_sample_factor)

        self.in_linear = nn.Linear(height, width, bias=False)

    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, -1)
        return x

class DenseDown2D(nn.Module):
    def __init__(self, signal_dim = (64, 128), down_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim[0] * signal_dim[1] * in_filters
        self.down_sample_factor = down_sample_factor
        width = int(signal_dim[0] * signal_dim[1] * n_filters / (self.down_sample_factor **2))

        self.in_linear = nn.Linear(height, width, bias=False)
        self.out_dim = (signal_dim[0] //  down_sample_factor, signal_dim[1] // down_sample_factor)
    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, self.out_dim[0], self.out_dim[1])

        return x

class DenseUp1D(nn.Module):
    def __init__(self, signal_dim = 128, up_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim * in_filters
        self.up_sample_factor = up_sample_factor
        width = int(signal_dim * n_filters * self.up_sample_factor)

        self.in_linear = nn.Linear(height, width, bias=False)

    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, -1)
        return x

class DenseUp2D(nn.Module):
    def __init__(self, signal_dim = (64, 128), up_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim[0] * signal_dim[1] * in_filters
        self.up_sample_factor = up_sample_factor
        width = int(signal_dim[0] * signal_dim[1] * n_filters * (self.up_sample_factor **2))

        self.in_linear = nn.Linear(height, width, bias=False)
        self.out_dim = (signal_dim[0] *  up_sample_factor, signal_dim[1] * up_sample_factor)
    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, self.out_dim[0], self.out_dim[1])

        return x

class DenseConv1D(nn.Module):
    def __init__(self, signal_dim = 128, up_layers = 4, in_filters = 4, mid_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        down_sample_factor = 2** up_layers
        n_filters = 2 ** up_layers * out_channels

        self.dense_layer = DenseDown1D(signal_dim, down_sample_factor, in_filters, mid_channels)
        self.conv_mid = DoubleConv1D(mid_channels, n_filters, n_filters // 2, kernel_size, padding, dilation)

        upsample_layers = []
        for u in range(up_layers):
            up = Up1D(out_channels * 2 ** (up_layers - u), out_channels * 2 **(up_layers -u - 1), kernel_size, padding, dilation)
            upsample_layers.append(up)
        
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv_out = DoubleConv1D(out_channels, out_channels, n_filters // 2, kernel_size, padding, dilation, True)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.conv_mid(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return x

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator1D(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm1d, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator1D, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock1D(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias )  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock1D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock1D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlock1D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlock1D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        self.model = UnetSkipConnectionBlock1D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock1D(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm1d, use_bias = True, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock1D, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias, padding_mode = 'circular')
        downrelu = nn.ReLU( True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2,True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose1d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGeneratorVH(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorVH, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlockV(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias)  # add the innermost layer
        for i in range(num_downs):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlockV(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias)

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlockH(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias)

        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlockH(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlockH(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)
        unet_block = UnetSkipConnectionBlockH(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias)

        self.model = UnetSkipConnectionBlockH(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlockV(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockV, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=(1, 4),
                             stride=(1, 2), padding=(0, 1), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(1,4), stride=(1, 2),
                                        padding=(0, 1), bias = use_bias)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=(1, 4), stride=(1, 2),
                                        padding= (0, 1), bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(1, 4), stride=(1, 2),
                                        padding=(0, 1), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlockH(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockH, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=(4,1),
                             stride=(2,1), padding=(1,0), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(4,1), stride=(2,1),
                                        padding=(1,0), bias = use_bias)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=(4,1), stride=(2,1),
                                        padding= (1,0), bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(4,1), stride=(2,1),
                                        padding=(1,0), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


if __name__ == '__main__':
    device = torch.device('cpu')
    # myconv = UNet2D(1, 1, mid_channels = 16, depth = 6, kernel_size= 3, padding = 2, dilation = 2, bias = True, sig_layer = False) #DoubleConv2D(1, 1)#
    model = UnetGeneratorVH(1, 1, 7)
    summary(model, (1, 256, 512))

    mse = nn.MSELoss()
    def myloss(output, target):
        return torch.log(mse(output, target))
    
    # myimg_path = '../cloud_data/barbara.png'
    # myimg = Image.open(myimg_path)
    # # myimg.show()
    # print(myimg)
    # # display()
    # myimg_tensor = transforms.ToTensor()(myimg).unsqueeze(0).type(torch.float)
    # print(myimg_tensor.size())
    # print('max in myimg: ', torch.max(myimg_tensor))
    # print('min in myimg: ', torch.min(myimg_tensor))

    # myimg_pil = transforms.ToPILImage()(myimg_tensor.squeeze(0))
    # myimg_pil.show()

    # myconv = myconv.to(device)
    # myinput = myimg_tensor.to(device)#torch.rand(1, 1, 512, 512).to(device)
    # mytarget = myimg_tensor.to(device) # torch.rand(1, 1, 256, 512).to(device)
    # # summary(myconv, (1, 256, 512))
    
    # myconv.train()
    # num_iter = 100
    # check_every = 10
    # print_every = 10
    # print(len(list(myconv.parameters())))
    # # curr_params = copy.deepcopy(list(myconv.parameters())) #and b = list(model.parameters())
    # curr_states = copy.deepcopy(myconv.state_dict())
    # myoptimizer = optim.Adam(myconv.parameters(), lr = 0.1)

    # for i in range(num_iter):
    #     # param = myconv.parameters()
    #     # old_gradients = [param.grad for param in myconv.parameters()]
    #     # for j in range(len(list(myconv.parameters()))):
    #     #     old.gradient
    #     myoutput = myconv(myinput)
    #     loss = mse(myoutput, mytarget)
    #     myoptimizer.zero_grad()
    #     loss.backward()
    #     myoptimizer.step()
        
    #     if i % print_every == 0:
    #         print(i, 'loss', loss)
    #         curr_output = myoutput.detach()
    #         curr_img = transforms.ToPILImage()(curr_output.squeeze(0))
    #         print('max in curr img: ', torch.max(curr_output))
    #         print('min in curr img: ', torch.min(curr_output))
    #         curr_img.show()

        
    #     if i % check_every == 0:
    #         everything_is_good = True
    #         new_gradients = [param.grad for param in myconv.parameters()]
    #         # print('old gradients:', old_gradients)
    #         # print('new gradients:', new_gradients)
    #         for grad in new_gradients:
    #             if grad == None:
    #                 raise ValueError('found grad none')
    #                 # everything_is_good = False

    #         new_states = copy.deepcopy(myconv.state_dict())
    #         for state_name in new_states:
    #             old_state = curr_states[state_name]
    #             new_state = new_states[state_name]
    #             if torch.equal(old_state, new_state):
    #                 print(state_name, 'not changed')
    #                 everything_is_good = False
    #         print('everything is good?', everything_is_good)
    #         # curr_params = new_params

    
    
