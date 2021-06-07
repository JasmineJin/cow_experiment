
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
    parser = argparse.ArgumentParser(description='parse arguments for testing')
    parser.add_argument('--data_directory', nargs='+', default=['cloud_data', 'mooooo', 'debug'])
    parser.add_argument('--model_path',  type = str, default = 'trained_model.pt')
    parser.add_argument('--save_path', type = str, default = 'roc.npz')
    parser.add_argument('--mode',  type = str, default = 'baseline')
    parser.add_argument('--show_every', type = int, default = 100)

    args = parser.parse_args()

    data_dir = os.path.join(*args.data_directory) 
    print('testing on data from directory: ', data_dir)

    extent = [-1, 1, 0, gen.max_rng]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path 
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    if mode == 'model':
        print('loaded model: ' + model_path)
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)[0:-1]

    show_figs = True
    check_all = False

    show_every = args.show_every
    use_baseline = args.mode == 'baseline'
    
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
                net_output_np = net_output.detach().numpy()[0, 0, :, :]
            
            net_output_np = gen.norm01_2d(net_output_np)
            target_np = target.detach().numpy()[0, 0, :, :]

            target_thresholded = target_np > 0.9
            output_thresholded = net_output_np > eta

            error = target_np - net_output_np

            detection  = output_thresholded * target_thresholded
            false_alarm = output_thresholded * (1 - target_thresholded)
            probability_false_alarm = np.sum(np.sum(false_alarm)) / np.sum(np.sum(1 - target_thresholded)) #* 100
            probability_detection = np.sum(np.sum(detection)) / np.sum(np.sum(target_thresholded)) #* 100
            average_probability_detection += probability_detection
            average_probability_false_alarm += probability_false_alarm
            nums_examined += 1
            if nums_examined % show_every == 0:
                print(sample['file_path'])

                fig, axs = plt.subplots(2, 2)
                pos0 = axs[0, 0].imshow(net_output_np, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
                axs[0, 0].set_title('net output')
                axs[0, 0].set_xlabel('cos(AoA)')
                axs[0, 0].set_ylabel('range (m)')
                
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
    
    plt.figure()
    plt.title('ROC')
    plt.plot(pfa_vector, pd_vector)
    plt.xlabel('PFA')
    plt.ylabel('PD')
    plt.show()

    np.savez_compressed(args.save_path, 
                    pf = pfa_vector,
                    pd = pd_vector)