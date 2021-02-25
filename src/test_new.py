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
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import datagen
from numpy.fft import fft, ifft
import newmodels

if __name__ == '__main__':
    print('finished importing stuff')
    device = torch.device('cpu')
    data_dir = os.path.join('cloud_data', 'vline', 'val')
    data_list = os.listdir(data_dir)[8:18]
    model_path = 'vlineA_fakegan_mini.pt'
    # model_path = os.path.join('models_trained', 'point_model2d_final.pt')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    print('loaded model')
    # print(model)

    # data_dir = os.path.join('../cloud_data', 'points', 'train')
    net_input_name = 'log_partial'
    target_name = 'log_full'
    # data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)

    show_figs = True
    nums_examine = 5
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')
    # print('cool')
    mydataset = mydata.PointDataSet(data_dir, data_list, net_input_name, target_name)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers=1)
    print('cool')

    for batch_idx, sample in enumerate(mydataloader):
        print('started loading')
        for name in sample:
            print(name)
            thing = sample[name]
        nums_examined += 1
        print(nums_examined)
        if show_figs:
            print('showing ', sample['file_path'])
            target = sample[target_name]
            target = mydata.norm01(target)
            target.to(device)
            net_input = sample[net_input_name]
            net_input = mydata.norm01(net_input)
            net_input.to(device)
            net_output = model(net_input)
            # print(sample['x_points'])
            # print(sample['y_points'])

            # mydata.display_data(target, net_output, net_input, target_name, net_input_name)
            # plt.figure()
            inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
            mydata.matplotlib_imshow(inputgrid, 'input')
            plt.figure()
            img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
            mydata.matplotlib_imshow(img_grid, 'output and target')
            # print(inputgrid.size())
            plt.show()
            # plt.show()
        
        if nums_examined >= nums_examine:
            break