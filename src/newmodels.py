import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import os
import copy
from torchsummary import summary
import queue
import copy
from PIL import Image
from torchvision import transforms


class MultiFilterDown(nn.Module):
    def __init__(self, in_channels, mid_channels, depth, use_bias = True):
        super(MultiFilterDown, self).__init__()

        conv_3x3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )
        conv_5x5 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 5, dilation = 2, padding = 4, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )
        layer3x3 = [conv_3x3]
        layer5x5 = [conv_5x5]
        for i in range(depth):
            layer3x3 +=[ nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )]
            layer5x5 += [nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size = 5, dilation = 2, padding = 4, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )]
        self.layer3x3 = nn.Sequential(*layer3x3)
        self.layer5x5 = nn.Sequential(*layer5x5)
    
    def forward(self, x):
        return torch.cat([self.layer3x3(x), self.layer5x5(x)], 1)

class MultifilterSame(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth, use_bias = True, last_act = nn.Sigmoid, use_dropout = False):
        super(MultifilterSame, self).__init__()

        conv_3x3 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            # nn.MaxPool2d(2),
                        )
        conv_5x5 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 5, dilation = 2, padding = 4, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            # nn.MaxPool2d(2),
                        )
        layer3x3 = [conv_3x3]
        layer5x5 = [conv_5x5]
        for i in range(depth):
            layer3x3 +=[ nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            # nn.MaxPool2d(2),
                        )]
            layer5x5 += [nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size = 5, dilation = 2, padding = 4, padding_mode = 'circular', bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            # nn.MaxPool2d(2),
                        )]
        if use_dropout:
            self.last_conv = nn.Sequential(nn.Dropout(p = 0.2),
                                nn.Conv2d(mid_channels * 2, out_channels, kernel_size = 3, dilation = 1, padding = 1, padding_mode = 'circular', bias = use_bias), 
                                nn.BatchNorm2d(out_channels),
                                last_act(),
                                # nn.MaxPool2d(2),
                            )
        else:
            self.last_conv = nn.Sequential(
                                nn.Conv2d(mid_channels * 2, out_channels, kernel_size = 3, dilation = 1, padding = 1, padding_mode = 'circular', bias = use_bias), 
                                nn.BatchNorm2d(out_channels),
                                last_act(),
                                # nn.MaxPool2d(2),
                            )
        # last_act = nn.Sigmoid
        self.layer3x3 = nn.Sequential(*layer3x3)
        self.layer5x5 = nn.Sequential(*layer5x5)
    
    def forward(self, x):
        x1 = torch.cat([self.layer3x3(x), self.layer5x5(x)], 1)
        return self.last_conv(x1)

class MultiFilterUp(nn.Module):
    def __init__(self, mid_channels, out_channels, depth, use_bias = True, last_act = nn.Sigmoid, use_dropout = False):
        super(MultiFilterUp, self).__init__()
        up_conv_layer = []
        for i in range(depth):
            up_conv_layer += [ nn.Sequential(nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size = 4, dilation = 1, stride = 2, padding = 1, bias = use_bias), 
                            nn.BatchNorm2d(mid_channels),
                            nn.ReLU(),
                            # nn.MaxPool2d(2),
                        )]
        if use_dropout:
            up_conv_layer += [nn.Dropout(p = 0.2)]
        up_conv_layer += [ nn.Sequential(nn.ConvTranspose2d(mid_channels, out_channels, kernel_size = 4, dilation = 1, stride = 2, padding = 1, bias = use_bias), 
                            nn.BatchNorm2d(out_channels),
                            last_act(),
                            # nn.MaxPool2d(2),
                        )]
        self.upconv = nn.Sequential(*up_conv_layer)
    def forward(self, x):
        return self.upconv(x)

class MultiFilter(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, depth, use_bias = True, last_act = nn.ReLU, use_dropout = False):
        super(MultiFilter, self).__init__()
        self.down = MultiFilterDown(in_channels, mid_channels, depth, use_bias)
        self.up = MultiFilterUp(mid_channels * 2, out_channels, depth, use_bias, last_act, use_dropout)
        # self.model = nn.Sequential(self.down, self.up)
    
    def forward(self, x):
        x = self.down(x)
        # print('mean in features: ', torch.mean(torch.abs(x)))
        # print('feature size: ', x.size())
        return self.up(x)

if __name__ == '__main__':
    # mymodel = Critic(1, 4, 128)
    mymodel = MultiFilter(1, 1, 64, 6)
    summary(mymodel.down, (1, 512, 1024))

    x = torch.rand(1, 1, 512, 1024)
    y = mymodel.down(x)
    print(y.size())