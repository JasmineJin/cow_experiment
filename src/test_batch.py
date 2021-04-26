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
    # model_path = 'single_point1000x100_autoencoder_newmodel_small_polar.pt'
    # model_path = 'points_polar_phase_mag_phase_polar_small_bce.pt' # last trained on 1000 things just points see 3/26
    model_path = 'points_polar_phase_10000x100_mag_phase_polar_big.pt' # last trained on 10000 things, see 4/5
    # model_path = 'points_polar_phase_10000x100_mag_phase_polar_big_bce.pt' # same as the above but bce loss not sure how many epochs i trained this on
    # model_path = 'fine_tune_polar_phase_fine_tune_pre_trained0.pt' # fine tuned with hard examples
    # model_path = os.path.join('models_trained', 'point_model2d_final.pt')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # model.train()
    print('loaded model: ' + model_path)
    # print(model)

    # data_dir = os.path.join('../cloud_data', 'points', 'train')
    # net_input_name = 'polar_partial2d_q1'
    # target_name = 'polar_full2d_q1'
    # data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
# import matplotlib.pyplot as plt
    # data_dir = os.path.join('cloud_data', 'mooooo', 'debug') # half points half vline
    # data_dir = os.path.join('cloud_data', 'points', 'test') # just points
    data_dir = os.path.join('cloud_data', 'testing', 'single_points')
    print('data directory: ', data_dir)
    # data_dir = os.path.join('cloud_data', 'testing', 'just_points') # a vline with a point next to it
    # data_dir = os.path.join('cloud_data', 'testing', 'mixed_stuff')
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)[0:-1]
    # ################################################################################
    # # check data statistics
    # ################################################################################
    # filepath = os.path.join(data_dir, data_list[5])
    # npzfile = np.load(filepath)
    # x_points = npzfile['all_point_x']
    # y_points = npzfile['all_point_y']
    # print('x_points: ', x_points)
    # print('y_points: ', y_points)
    # raw_data = datagen.get_scene_raw_data(x_points, y_points)
    
    # print('max magnitude: ', np.max(np.max(np.abs(raw_data))))
    # print('min magnitude: ', np.min(np.min(np.abs(raw_data))))

    # plt.figure()
    # plt.plot(np.imag(raw_data[10, :]))
    # plt.title('raw data for at one time')
    # plt.show()

    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
    ###############################################################################
    # check and plot data from thing
    ###############################################################################
    show_figs = True
    check_all = False
    nums_examine = 100
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')
    bce = nn.BCELoss(reduction = 'mean')
    mydataset = mydata.PointDataSet(data_dir, data_list, pre_processed = True)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    total_error = 0
    for batch_idx, sample in enumerate(mydataloader):
        # nums_examined += 1
        # print(nums_examined)
        
        
        target = sample[target_name]
        target = mydata.norm01(target)
        net_input = sample[net_input_name]
        # net_input = mydata.norm01(net_input)
        net_output = model(net_input)

        # print(sample['x_points'])
        # print(sample['y_points'])
        # display_data(target, target, net_input, target_name, net_input_name)
        loss = mse(net_output, target)
        # loss = loss / torch.sum(torch.abs(target))
        
        total_error += loss.item()
        # plt.figure()
        if nums_examined % 20 == 0:
            print(sample['file_path'])
            print(loss)

            mydata.plot_labeled_grid(net_input, net_output, target)
            # inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
            # mydata.matplotlib_imshow(inputgrid, 'input')
            # plt.figure()
            # img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
            # mydata.matplotlib_imshow(img_grid, 'output and target')
            # print(inputgrid.size())
            plt.show()
        nums_examined += 1
        # if nums_examined >= nums_examine:
        #     break

    print('average se: ', total_error / nums_examine)