# Blessed Carlo Acutis, pray for us
# St. Thomas Aquinas, pray for us
# St. Ignatius Loyola, pray for us
# St. Joseph, pray for us
# St. Jude, pray for us
# Holy Mary, Seat of Wisdom, pray for us
import torch
import json
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

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
import image_transformation_utils as trans

if __name__ == '__main__':
    data_dir = os.path.join('cloud_data', 'mooooo', 'debug')
    # data_dir = os.path.join('cloud_data', 'points', 'test')
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)[1:-1]
    
    filepath = os.path.join(data_dir, data_list[0])
    # mydata['file_path'] = filepath
    npzfile = np.load(filepath)
    x_points = npzfile['all_point_x']
    # print(x_points)
    y_points = npzfile['all_point_y']

    gt = gen.get_ground_truth(x_points, y_points)

    plt.figure()
    plt.imshow(gt)
    plt.title('ground truth')
    plt.show()