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
# from numpy.fft import fft, ifft

if __name__ == '__main__':
    print('finished importing stuff')
    device = torch.device('cpu')
    # model_path = 'single_point1000x100_autoencoder_newmodel_small_polar.pt'
    # model_path = 'points_polar_phase_mag_phase_polar_small_bce.pt' # last trained on 1000 things just points see 3/26
    model_path = 'points_polar_phase_10000x100_mag_phase_polar_big.pt' # last trained on 10000 things, see 4/5
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
    # data_dir = os.path.join('cloud_data', 'mooooo', 'debug')
    data_dir = os.path.join('cloud_data', 'points', 'test')
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)[1:-1]
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
    nums_examine = 1
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'mean')
    bce = nn.BCELoss(reduction = 'mean')
    mydataset = mydata.PointDataSet(data_dir, data_list, pre_processed = False)
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
            # net_input = mydata.norm01(net_input)
            net_output = model(net_input)

            # print(sample['x_points'])
            # print(sample['y_points'])
            # display_data(target, target, net_input, target_name, net_input_name)
            print(mse(net_output, target))
            # plt.figure()
            # inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
            # mydata.matplotlib_imshow(inputgrid, 'input')
            # plt.figure()
            # img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
            # mydata.matplotlib_imshow(img_grid, 'output and target')
            # print(inputgrid.size())
            # plt.show()
        
        if nums_examined >= nums_examine:
            break

    #####################################################################################
    # check physics
    #####################################################################################
    print(sample['file_path'])
    net_output = mydata.norm01(net_output)
    output_np = net_output.detach().numpy()
    output_np = output_np[0, 0, :, :]
    print('output size: ', output_np.shape)
    print('output max: ', np.max(np.max(output_np)))
    print('output min: ', np.min(np.min(output_np)))
    my_target = sample[target_name]
    my_target_np = my_target.detach().numpy()
    my_target_np = my_target_np[0, 0, :, :]
    print('target size: ', my_target_np.shape)
    print('target max: ', np.max(np.max(my_target_np)))
    print('target min: ', np.min(np.min(my_target_np)))

    # output_thresholded = output_np * (output_np > 0.6)
    # apply_threshold_per_row
    thresholding_mtx = gen.apply_threshold_per_row(output_np, np.max(np.max(output_np))* 0.2, np.max(np.max(output_np)) * 0.4)
    # plt.figure()
    output_thresholded = output_np * thresholding_mtx

    plt.figure()
    plt.imshow(output_thresholded)
    plt.title('output image thresholded')
    plt.show()

    output_norm2target = output_thresholded * (np.max(np.max(my_target_np)) - np.min(np.min(my_target_np))) + np.min(np.min(my_target_np))
    print('re-normalized output min: ', np.min(np.min(output_norm2target)))
    print('re-normalized output max: ', np.max(np.max(output_norm2target)))


    output_exp = np.exp(output_norm2target * np.log(10))
    plt.figure()
    plt.imshow(output_exp)
    plt.title('output image exponentiated')
    plt.show()

    actual_raw_data = sample['raw_data']
    actual_raw_data = actual_raw_data.numpy()[0, :, :]

    
    print('actual raw data shape: ', actual_raw_data.shape)
    # range_processed = fft(np.dot(range_window_mtx, array_response), axis = 0, norm= 'ortho')
    angle_unprocessed = ifft(output_exp, n = gen.num_channels, axis= 1, norm = 'ortho')
    range_unprocessed = ifft(angle_unprocessed, n = gen.num_channels, axis= 0, norm = 'ortho')
    # print("my output exponentiated")
    print('fake raw data shape: ', range_unprocessed.shape)

    actual_raw_data_squared = np.abs(actual_raw_data) * np.abs(actual_raw_data)
    fake_raw_data_squared = np.abs(range_unprocessed) * np.abs(range_unprocessed)
    print('actual raw data energy: ', np.sum(np.sum(actual_raw_data_squared)))
    print('fake raw data energy: ', np.sum(np.sum(fake_raw_data_squared)))

    
    # output_thresholded
    # rng_vector = np.arange(gen.num_range_bins) * gen.rng_res
    # output_rect, x, y = trans.polar_to_rect(output_thresholded, gen.wl, gen.num_channels, rng_vector, 512, 1024)
    # plt.figure()
    # plt.imshow(output_rect)
    # plt.title('output image normal coord')
    # plt.show()

    # inputgrid = mydata.get_input_image_grid(net_input, net_input_name)
    # mydata.matplotlib_imshow(inputgrid, 'input')
    # plt.figure()
    # img_grid = mydata.get_output_target_image_grid(net_output, target, target_name)
    # mydata.matplotlib_imshow(img_grid, 'output and target')
    # # print(inputgrid.size())
    # plt.show()
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