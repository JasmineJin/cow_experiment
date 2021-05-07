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
    # model_path = 'points_polar_phase_mag_phase_polar_small_bce.pt'
    model_path = 'vline_cnn_mini.pt'
    # model_path = os.path.join('models_trained', 'point_model2d_final.pt')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # model.train()
    print('loaded model')
    # print(model)

    # data_dir = os.path.join('../cloud_data', 'points', 'train')
    # net_input_name = 'polar_partial2d_q1'
    # target_name = 'polar_full2d_q1'
    # data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
# import matplotlib.pyplot as plt
    # data_dir = os.path.join('cloud_data', 'vline', 'debug')
    data_dir = os.path.join('cloud_data', 'testing', 'mixed_stuff')
    net_input_name = 'log_partial'
    target_name = 'log_full'
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

    data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
    ###############################################################################
    # check and plot data from thing
    ###############################################################################
    show_figs = True
    check_all = False
    nums_examine = 5
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')
    bce = nn.BCELoss(reduction = 'mean')
    mydataset = mydata.PointDataSet(data_dir, data_list, net_input_name, target_name, pre_processed = False)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    
    for batch_idx, sample in enumerate(mydataloader):
        # for name in sample:
        #     print(name)
        #     thing = sample[name]
        #     if name == net_input_name or name == target_name:
        #         # print(name)
        #         print('size: ', thing.size())
        #         print('minimum value: ', torch.min(thing))
        #         print('maximum value: ', torch.max(thing))
        #         print('average value: ', torch.mean(thing))
        #         print('variance: ', torch.var(thing))
        #         zero_tensor = torch.zeros(thing.size())
        #         # print('0:', torch.sum(zero_tensor))
        #         mynorm = mse(thing, zero_tensor)
        #         print('sum abs squared: ', mynorm.item())
        #     else:
        #         print(thing)
        nums_examined += 1
        print(nums_examined)
        if show_figs:
            target = sample[target_name]
            target = mydata.norm01(target)
            net_input = sample[net_input_name]
            net_input = mydata.norm01(net_input)
            net_output = model(net_input)

            # print(sample['x_points'])
            # print(sample['y_points'])
            mydata.display_data(target, net_output, net_input, target_name, net_input_name)
            print(mse(net_output, target))
            # plt.figure()
            # inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
            # mydata.matplotlib_imshow(inputgrid, 'input')
            # plt.figure()
            # img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
            # mydata.matplotlib_imshow(img_grid, 'output and target')
            # print(inputgrid.size())
            plt.show()
        
        if nums_examined >= nums_examine:
            break

        ##################################################################################
        # check the data process
        ##################################################################################
    
    # #######################################################
    # # test quantizer
    # #######################################################
    # mydata = np.linspace(-1, 11, num = 500)
    # q_mydata = datagen.quantizer(mydata, low = 0, high = 9, n_levels = 5)

    # plt.figure()
    # plt.plot(mydata, q_mydata)
    # plt.show()

    # show_figs = True
    # nums_examine = 10
    # nums_examined = 0

    # # mse = nn.MSELoss(reduction = 'sum')
    # # print('cool')
    # mydataset = mydata.PointDataSet(data_dir, data_list, net_input_name, target_name)
    # mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers=1)
    # print('cool')

    # for batch_idx, sample in enumerate(mydataloader):
    #     print('started loading')
    #     for name in sample:
    #         print(name)
    #         thing = sample[name]
    #     nums_examined += 1
    #     print(nums_examined)
    #     if show_figs:
    #         print('showing ', sample['file_path'])
    #         target = sample[target_name]
    #         target = mydata.norm01(target)
    #         target.to(device)
    #         # net_input = sample[target_name]
    #         # net_input.to(device)
    #         net_output = model(target)
    #         # print(sample['x_points'])
    #         # print(sample['y_points'])

    #         # mydata.display_data(target, net_output, net_input, target_name, net_input_name)
    #         # plt.figure()
    #         # inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
    #         # mydata.matplotlib_imshow(inputgrid, 'input')
    #         plt.figure()
    #         img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
    #         mydata.matplotlib_imshow(img_grid, 'output and target')
    #         # print(inputgrid.size())
    #         plt.show()
    #         # plt.show()
        
    #     if nums_examined >= nums_examine:
    #         break