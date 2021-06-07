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

# import models
import newmodels
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import data_generation as gen
from numpy.fft import fft, ifft

if __name__ == '__main__':
    print('finished importing stuff')
    extent = [-1, 1, 0, gen.max_rng]
    device = torch.device('cpu')
    # model_path = 'single_point1000x100_autoencoder_newmodel_small_polar.pt'
    # model_path = 'points_polar_phase_mag_phase_polar_small_bce.pt' # last trained on 1000 things just points see 3/26
    model_path = 'points_polar_phase_10000x100_mag_phase_polar_big.pt' # last trained on 10000 things, see 4/5
    # model_path = 'points_polar_phase_10000x100_mag_phase_polar_big_bce.pt' # same as the above but bce loss not sure how many epochs i trained this on
    # model_path = 'fine_tune_polar_phase_fine_tune_pre_trained0.pt' # fine tuned with hard examples
    # model_path = 'norm_mse_10000x5_mag_phase_polar_big.pt'
    model = torch.load(model_path, map_location=device)
    model.to(device)
    # model.train()
    print('loaded model: ' + model_path)
    # summary(model, (4, 512, 1024))
    # print(model)

    # data_dir = os.path.join('../cloud_data', 'points', 'train')
    # net_input_name = 'polar_partial2d_q1'
    # target_name = 'polar_full2d_q1'
    # data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)
    # data_dir = os.path.join('cloud_data', 'mooooo', 'debug') # half points half vline
    # data_dir = os.path.join('cloud_data', 'points', 'test') # just points
    # data_dir = os.path.join('cloud_data', 'cluster', 'test') # just points
    # data_dir = os.path.join('cloud_data', 'testing', 'single_points')
    data_dir = os.path.join('cloud_data', 'mooooo', 'debug') # a vline with a point next to it
    # data_dir = os.path.join('cloud_data', 'testing', 'mixed_stuff')
    print('data directory: ', data_dir)
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
        
    show_every = 100
    use_baseline = True
    mse = nn.MSELoss(reduction = 'sum')
    bce = nn.BCELoss(reduction = 'mean')
    mydataset = mydata.PointDataSet(data_dir, data_list, pre_processed = True)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    eta_vector = np.linspace(0, 1, num = 100)
    pd_vector = np.zeros(eta_vector.shape)
    pfa_vector = np.zeros(eta_vector.shape)
    total_error = 0
    for eta_idx in range(100):
        print('checking eta: ', eta_idx)
        eta = eta_vector[eta_idx]
        average_probability_detection = 0
        average_probability_false_alarm = 0
        nums_examined = 0
        for batch_idx, sample in enumerate(mydataloader):
            target = sample[target_name]
            target = mydata.norm01(target)
            net_input = sample[net_input_name]
            if use_baseline:
                net_input_np = net_input.detach().numpy()
                net_cos0 = net_input_np[0, 0, :, :]
                net_sin0 = net_input_np[0, 1, :, :]
                net_output_np = np.abs(net_cos0 + 1j * net_sin0)
            else:
                net_output = model(net_input)
                # net_output = mydata.norm01(net_output)
                net_output_np = net_output.detach().numpy()[0, 0, :, :]
            
            net_output_np = gen.norm01_2d(net_output_np)
            target_np = target.detach().numpy()[0, 0, :, :]

            # target_thresholded = gen.apply_threshold_per_row(target_np, 0.1, 0.3)
            target_thresholded = target_np > 0.9
            # plt.figure()
            # plt.imshow(target_thresholded)
            # plt.title('target detection')
            # output_thresholded = gen.apply_threshold_per_row(net_output_np, 0.3, 0.3)
            output_thresholded = net_output_np > eta
            # plt.figure()
            # plt.imshow(output_thresholded)
            # plt.title('output detection')
            # # loss = mse(net_output, target)
            # plt.show()
            error = target_np - net_output_np
            # mse = np.sum(np.sum(error * error))
            # total_error += mse
            detection  = output_thresholded * target_thresholded
            false_alarm = output_thresholded * (1 - target_thresholded)
            probability_false_alarm = np.sum(np.sum(false_alarm)) / np.sum(np.sum(1 - target_thresholded)) #* 100
            probability_detection = np.sum(np.sum(detection)) / np.sum(np.sum(target_thresholded)) #* 100
            average_probability_detection += probability_detection
            average_probability_false_alarm += probability_false_alarm
            nums_examined += 1
            if nums_examined % show_every == 0:
                print(sample['file_path'])
                # print(loss)

                # mydata.plot_labeled_grid(net_input, net_output, target)
                # plt.show()
                fig, axs = plt.subplots(2, 2)
                pos0 = axs[0, 0].imshow(net_output_np, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
                axs[0, 0].set_title('net output')
                axs[0, 0].set_xlabel('cos(AoA)')
                axs[0, 0].set_ylabel('range (m)')
                # cbar = fig.colorbar(pos0)
                pos1 = axs[0, 1].imshow(target_np, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
                axs[0, 1].set_title('target')
                axs[0, 1].set_xlabel('cos(AoA)')
                axs[0, 1].set_ylabel('range (m)')
                pos3 = axs[1,0].imshow(output_thresholded, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
                axs[1,0].set_title('Prediction')
                axs[1,0].set_xlabel('cos(AoA)')
                axs[1,0].set_ylabel('range (m)')
                pos4 = axs[1,1].imshow(target_thresholded, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
                axs[1,1].set_title('Target')
                axs[1,1].set_xlabel('cos(AoA)')
                axs[1,1].set_ylabel('range (m)')
                print('pd: ', probability_detection)
                print('pfa: ', probability_false_alarm)
                plt.show()
        pd_vector[eta_idx] = average_probability_detection
        pfa_vector[eta_idx] = average_probability_false_alarm

        # break
    
    plt.figure()
    plt.title('ROC')
    plt.plot(pfa_vector, pd_vector)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.show()

    np.savez_compressed('roc_big_points.npz', 
                    pf = pfa_vector,
                    pd = pd_vector)