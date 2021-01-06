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

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = (3, 3), padding = (2, 2), dilation = (1, 1), final_layer = False):
        super().__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
            )
            conv_layers.append(double_conv1)
        else:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            conv_layers.append(double_conv1)
        self.double_conv = nn.Sequential(*conv_layers)
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,  kernel_size = 3, padding = 2, dilation = 1, final_layer = False):
        super().__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                nn.BatchNorm1d(out_channels),
                nn.Sigmoid(),
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
        super().__init__()
        self.conv = DoubleConv1D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)

    def forward(self, x1):
        x1 = F.interpolate(x1, scale_factor= 2)

        # x = torch.cat([x1, x2], dim = 1)
        # x = x1 + x2
        x = x1
        return self.conv(x)

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)

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

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)
        
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu')):
        super().__init__()
        self.down_layers = []
        self.up_layers = []

        self.inlayer = DoubleConv1D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation)
        # self.midlayer = DoubleConv1D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.outlayer = DoubleConv1D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation)

        for d in range(depth):
            downlayer = Down1D(mid_channels * 2 ** d, mid_channels * (2** (d + 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.down_layers.append(downlayer.to(device))
        
        for d in range(depth):
            # print(d)
            uplayer = Up1D(mid_channels * (2 ** (depth - d)), mid_channels * (2**(depth - d - 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.up_layers.append(uplayer.to(device))

    def forward(self, x):
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
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu') , sig_layer = True):
        super().__init__()
        self.down_layers = []
        self.up_layers = []

        self.inlayer = DoubleConv2D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation).to(device)
        # self.midlayer = DoubleConv2D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.outlayer = DoubleConv2D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, final_layer= sig_layer).to(device)

        for d in range(depth):
            downlayer = Down2D(mid_channels * 2 ** d, mid_channels * (2** (d + 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.down_layers.append(downlayer.to(device))
        
        for d in range(depth):
            # print(d)
            uplayer = Up2D(mid_channels * (2 ** (depth - d)), mid_channels * (2**(depth - d - 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.up_layers.append(uplayer.to(device))

    def forward(self, x):
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


if __name__ == '__main__':
    
    myconv1d = DoubleConv2D(2, 4, kernel_size= 3, padding = 1, dilation = 1)
    # myconv2d = DoubleConv2D(2, 4, kernel_size= 3, padding = 2, dilation = 1)
    summary(myconv1d, (2, 64, 128))
    # summary(myconv2d, (2, 64, 128))

    # mydown1d = Down1D(2, 4, kernel_size=3, padding=2, dilation=1)
    # mydown2d = Down2D(2, 4, kernel_size=3, padding=2, dilation=1)
    # summary(mydown1d, (2, 128))
    # summary(mydown2d, (2, 64, 128))

    # myup1d = Up1D(2, 4, kernel_size=3, padding=2, dilation=1)
    # myup2d = Up2D(2, 4, kernel_size=3, padding=2, dilation=1)
    # summary(myup1d, (2, 128))
    # summary(myup2d, (2, 64, 128))

    # myinput2d = torch.rand(1, 2, 32, 64)
    # # myoutput = myup1d(myinput)
    # # print(myoutput.size())

    # # device = 
    # # myunet = UNet2D(depth = 6)
    # densedown2d = DenseUp2D(signal_dim = (32,64), up_sample_factor= 4, in_filters= 2, n_filters = 1)
    # myoutput2d = densedown2d(myinput2d)
    # print(myoutput2d.size())
    # summary(myunet, (2, 128))

    # myinput1d = torch.rand(1, 2, 256)
    # densedown1d = DenseUp1D(signal_dim = 256, up_sample_factor= 4, in_filters= 2, n_filters = 4)
    # myoutput1d = densedown1d(myinput1d)
    # denseconv = DenseConv1D(signal_dim = 256, up_layers= 4, in_filters= 2, out_channels= 1, kernel_size= 3, padding = 2, dilation = 1)
    # myoutput1d = denseconv(myinput1d)
    # print(myoutput1d.size())