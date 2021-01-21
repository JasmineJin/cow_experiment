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

if __name__ == '__main__':
    print('finished importing stuff')
    device = torch.device('cpu')
    data_dir = os.path.join('..\cloud_data', 'points', 'train')
    data_list = os.listdir(data_dir)
    model_path = 'single_point_unetvh_small.pt'
    # model_path = os.path.join('models_trained', 'point_model2d_final.pt')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    print('loaded model')
    # print(model)

    data_dir = os.path.join('../cloud_data', 'points', 'train')
    net_input_name = 'polar_partial2d'
    target_name = 'polar_full2d'
    data_list = os.listdir(data_dir)[100:150]
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)

    show_figs = True
    nums_examine = 1
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')
    # print('cool')
    mydataset = mydata.PointDataSet(data_dir, data_list, net_input_name, target_name)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= True, num_workers=1)
    print('cool')

    for batch_idx, sample in enumerate(mydataloader):
        print('started loading')
        for name in sample:
            print(name)
            thing = sample[name]
            # if name == net_input_name or name == target_name:
            #     print('size: ', thing.size())
            #     print('minimum value: ', torch.min(thing))
            #     print('maximum value: ', torch.max(thing))
            #     print('average value: ', torch.mean(thing))
            #     print('variance: ', torch.var(thing))
            #     zero_tensor = torch.zeros(thing.size())
            #     # print('0:', torch.sum(zero_tensor))
            #     mynorm = mse(thing, zero_tensor)
            #     print('sum abs squared: ', mynorm.item())
            # else:
            #     print(thing)
        nums_examined += 1
        print(nums_examined)
        if show_figs:
            target = sample[target_name]
            target.to(device)
            net_input = sample[net_input_name]
            net_input.to(device)
            net_output = model(net_input)
            # print(sample['x_points'])
            # print(sample['y_points'])

            mydata.display_data(target, net_output, net_input, target_name, net_input_name)

            plt.show()
        
        if nums_examined >= nums_examine:
            break