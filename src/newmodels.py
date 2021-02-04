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
    mymodel = ConvDown(2, 1, 64, 6)

    summary(mymodel, (2, 256, 512))

    x = torch.rand(1, 2, 256, 512)
    y = mymodel(x)
    print(y)