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
import datagen

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
        # self.outconv = nn.Sequential(nn.Conv2d(2 * channels, mid_channels, kernel_size = 3, dilation = 2, padding = 4, padding_mode = 'circular'), 
        #                     nn.BatchNorm2d(mid_channels),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2),
        #                 )
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
        # self.outconv = nn.Sequential(nn.Conv2d(2 * channels, mid_channels, kernel_size = 3, dilation = 2, padding = 4, padding_mode = 'circular'), 
        #                     nn.BatchNorm2d(mid_channels),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2),
        #                 )
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

class Critic(nn.Module):
    def __init__(self, in_channels, mid_channels, h_dim, use_bias = True, last_act = nn.ReLU, use_dropout = False):
        super(Critic, self).__init__()
        self.down = MultiFilterDown(in_channels, mid_channels, 3, use_bias)
        self.up = nn.Sequential(nn.Linear(mid_channels* 2 * 16 * 32, h_dim, bias = use_bias),
                                nn.LeakyReLU(0.1),
                                nn.Linear(h_dim, 1, bias = use_bias),
                                nn.Tanh())
        # self.model = nn.Sequential(self.down, self.up)
    
    def forward(self, x):
        x = self.down(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print('mean in features: ', torch.mean(torch.abs(x)))
        # print('feature size: ', x.size())
        return self.up(x)

class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, depth, use_bias = True):
        super(ConvDown, self).__init__()
        conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular'), 
                            nn.BatchNorm2d(mid_channels),
                            nn.MaxPool2d(2),
                                    nn.ReLU(),
                        )
        layers = [conv1]
        for i in range(depth - 2):
            layers += [nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular'), 
                            nn.BatchNorm2d(mid_channels),
                            nn.MaxPool2d(2),
                                    nn.ReLU(),
                        )]
        layers += [nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size = 3, dilation = 2, padding = 2, padding_mode = 'circular'), 
                            nn.BatchNorm2d(out_channels),
                            nn.MaxPool2d(2),
                                    nn.ReLU(),
                        )]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)


class SuperDense(nn.Module):
    def __init__(self, depth = 4, out_dim = 16, use_bias = True):
        super(SuperDense, self).__init__()
        lin1 = nn.Sequential(nn.Linear(4 * datagen.num_samples, out_dim * 8, bias = use_bias), 
                                    nn.ReLU(),
                                    )
        lin2 = nn.Sequential(nn.Linear(out_dim * 8, out_dim * 4, bias = use_bias),
                                    nn.ReLU(),
                                    )
        lin3 = nn.Sequential(nn.Linear(out_dim * 4, out_dim * 2, bias = use_bias),
                                    nn.ReLU(),
                                    )
        lin4 = nn.Sequential(nn.Linear(out_dim * 2, out_dim * 1, bias = use_bias),
                                    nn.ReLU(),
                                    )
        my_layers = [lin1, lin2, lin3, lin4]
        for i in range(depth - 4):
            my_layers += [nn.Sequential(nn.Linear(out_dim, out_dim, bias = use_bias),
                            nn.ReLU(),
                            )]
        # my_layers += nn.Sequential(nn.Linear(out_dim, out_dim, bias = use_bias),
        #                             nn.Sigmoid(),
        #                             )
        self.model = nn.Sequential(*my_layers)
    def forward(self, x):
        return self.model(x)

class DenseConv(nn.Module):
    def __init__(self, nconv = 4):
        super(DenseConv, self).__init__()
        self.dense = SuperDense()
        conv_layers = []

        for n in range(nconv):
            conv_layers += [nn.Sequential(nn.Conv2d(1, 1, kernel_size = (3, 1), padding = (1, 0),stride = (2, 1)),
                            nn.ReLU(),
                            )]
            # [nn.Sequential(nn.Conv2d(1, 1, kernel_size = (3, 1), stride = (2, 1)), nn.ReLu(),)]

        self.conv_layers = nn.Sequential(*conv_layers)
    def forward(self, x):
        x = self.dense(x).unsqueeze(1)
        return self.conv_layers(x)
    # def forward(self, x):
    #      = self.dense(x)

class LocPredictor(nn.Module):
    def __init__(self):
        super(LocPredictor, self).__init__()
        self.encoder = ConvDown(2, 2, 64, 6)
        self.predictorx = nn.Sequential(
                                     nn.Linear(64, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, 1),)
        self.predictory = nn.Sequential(nn.Linear(64, 16),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(16, 1),
                                    nn.LeakyReLU(0.2),
                                    )
    def forward(self, x):
        code = self.encoder(x).view(x.size(0), -1)
        return self.predictorx(code), self.predictory(code)
        
if __name__ == '__main__':
    # mymodel = Critic(1, 4, 128)
    mymodel = MultiFilter(1, 1, 4, 4)
    summary(mymodel.down, (1, 256, 512))

    x = torch.rand(1, 1, 256, 512)
    y = mymodel.down(x)
    print(y.size())