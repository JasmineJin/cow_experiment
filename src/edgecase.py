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
import datagen_new as gen
from numpy.fft import fft, ifft

if __name__ == '__main__': 
    extent = [-1, 1, 0, gen.max_rng]

    r = 20
    point1_x = [0]
    point1_y = [20]

    raw_data1 = gen.get_scene_raw_data(point1_x, point1_y)
    processed1 = gen.get_radar_image_pairs(raw_data1)

    full1 = processed1['polar_full'][0, :, :]
    print(full1.shape)
    plt.figure()
    plt.imshow(full1, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.title('singe point at range 20m')

    angle = 0.1
    point2_x = [0, r* np.sin(angle)]
    point2_y = [20, r * np.cos(angle)]
    raw_data2 = gen.get_scene_raw_data(point2_x, point2_y)
    processed2 = gen.get_radar_image_pairs(raw_data2)

    full2 = processed2['polar_full'][0, :, :]
    # print(full1.shape)
    plt.figure()
    plt.imshow(full2, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.title('two point at range 20m')


    point3_x = [ r* np.sin(angle)]
    point3_y = [ r * np.cos(angle)]
    raw_data3 = gen.get_scene_raw_data(point3_x, point3_y)
    processed3 = gen.get_radar_image_pairs(raw_data3)

    full3 = processed3['polar_full'][0, :, :]


    raw_partial0_from_sample1 = raw_data1[:, 0: 12]
    raw_partial0_from_sample3 = raw_data3[:, 0: 12]
    diff = raw_partial0_from_sample1 - raw_partial0_from_sample3
    print('raw data difference', np.sum(np.sum(np.abs(diff))))

    p0 = gen.process_array(raw_partial0_from_sample1)
    p1 = gen.process_array(raw_partial0_from_sample3)

    abs_diff = np.abs(p1) - np.abs(p0)
    print('transform diff', np.sum(np.sum(np.abs(abs_diff))))
    # plt.show()

    log_0 = np.log10(np.abs(p0))
    log_1 = np.log10(np.abs(p1))

    log_diff = log_0 - log_1
    print('log diff total', np.sum(np.sum(np.abs(log_diff))))

    norm_log0 = (log_0 - np.min(np.min(log_0)))/ (np.max(np.max(log_0)) - np.min(np.min(log_0)))
    norm_log1 = (log_1 - np.min(np.min(log_1)))/ (np.max(np.max(log_1)) - np.min(np.min(log_1)))
    # norm_log0 = norm_log0[200:300, 400:600] 
    # norm_log1 = norm_log1[200:300, 400:600] 
    
    norm_log_diff= norm_log0 - norm_log1
    print('norm log diff total:', np.sum(np.sum(np.abs(norm_log_diff))))
    print('norm log diff average', np.mean(np.mean(np.abs(norm_log_diff))))
    print('norm log diff max', np.max(np.max(np.abs(norm_log_diff))))

    plt.figure()
    plt.imshow(norm_log0, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('Scene 1')
    plt.figure()
    plt.imshow(norm_log1, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.title('Scene 2')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.show()
    # print(np.min(np.min(norm_log0)))



