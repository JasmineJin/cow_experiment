import numpy as np 
import torch
import json
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import os
import argparse
from torch.optim import lr_scheduler
import torch.utils.data as data

import models
import newmodels
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import datagen
from numpy.fft import fft, ifft

if __name__ == '__main__':
    print('finished importing stuff')
    device = torch.device('cpu')
    model_path = 'single_point1000_newmodel_small_polar.pt'
    # model_path = os.path.join('models_trained', 'point_model2d_final.pt')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # model.train()
    print('loaded model')
    down = model.down
    up = model.up
    # print(up)

    random_input = torch.randn(1, 8, 4, 8)
    random_input =  nn.ReLU()(random_input)
    random_output = up(random_input)
    random_output_np = random_output.detach().numpy()
    print(random_output.size())
    plt.figure()
    plt.imshow(random_output_np[0, 0, :, :], cmap = 'gray')
    plt.title('decoder output with random input')
    plt.show()
    # data_dir = os.path.join('../cloud_data', 'points', 'train')
    # net_input_name = 'polar_partial2d_q1'
    # target_name = 'polar_full2d_q1'
    # data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
    # import matplotlib.pyplot as plt