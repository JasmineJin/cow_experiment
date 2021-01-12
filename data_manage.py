import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
import matplotlib.pyplot as plt 
import datagen
import image_transformation_utils as trans 

class PointDataSet(data.Dataset):
    def __init__(self, data_dir, data_list):
        self.data_dir = data_dir
        self.data_list = data_list
    def __len__(self):
        
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        mydata = {}
        filepath = os.path.join(self.data_dir, self.data_list[idx])
        npzfile = np.load(filepath)
        x_points = npzfile['all_point_x']
        # print(x_points)
        y_points = npzfile['all_point_y']
        mydata['x_points'] = x_points
        mydata['y_points'] = y_points
        raw_data = datagen.get_scene_raw_data(x_points, y_points)

        processed = datagen.get_radar_image_pairs(raw_data)

        target_np = 20 * np.log10(processed['mag_full'] + 10 **(-12))
        partial0_np = processed['mag_partial0']
        partial1_np = processed['mag_partial1']
        

        target_np = target_np[np.newaxis, :, :]
        target_torch = torch.from_numpy(target_np).type(torch.float)
        mydata['log_full'] = target_torch
        
        partial_np = np.dstack((partial0_np, partial1_np))
        partial_np = 20 * np.log10(partial_np + 10 ** (-12))
        partial_np = partial_np.transpose(2, 0, 1)
        partial_torch = torch.from_numpy(partial_np).type(torch.float)
        mydata['log_partial'] = partial_torch

        polar_full_np = processed['polar_full']
        polar_full_np = np.hstack([polar_full_np.real, polar_full_np.imag])
        polar_partial0_np = processed['polar_partial0']
        polar_partial1_np = processed['polar_partial1']
        polar_partial_np = np.hstack([polar_partial0_np.real, polar_partial0_np.imag, polar_partial1_np.real, polar_partial1_np.imag])

        polar_full_torch = torch.from_numpy(polar_full_np).type(torch.float)
        polar_partial_torch = torch.from_numpy(polar_partial_np).type(torch.float)
        # all_data = np.dstack((target_np, partial0_np, partial1_np))
        mydata['polar_full'] = polar_full_torch
        mydata['polar_partial'] = polar_partial_torch

        real_full_np = processed['real_full']
        imag_full_np = processed['imag_full']

        real_partial0_np = processed['real_partial0']
        imag_partial0_np = processed['imag_partial0']

        real_partial1_np = processed['real_partial1']
        imag_partial1_np = processed['imag_partial1']

        full_np = np.dstack((real_full_np, imag_full_np))
        full_np = full_np.transpose(2, 0, 1)
        partial_np = np.dstack((real_partial0_np, imag_partial0_np, real_partial1_np, imag_partial1_np))
        partial_np = partial_np.transpose(2, 0, 1)

        full_torch = torch.from_numpy(full_np).type(torch.float)
        partial_torch = torch.from_numpy(partial_np).type(torch.float)
        mydata['full'] = full_torch
        mydata['partial'] = partial_torch

        return mydata

class Point1DDataSet(data.Dataset):
    def __init__(self, data_dir, data_list):
        self.data_dir = data_dir
        self.data_list = data_list
    def __len__(self):
        
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        mydata = {}
        filepath = os.path.join(self.data_dir, self.data_list[idx])
        npzfile = np.load(filepath)

        target_np = npzfile['target']
        net_input_np = npzfile['net_input']

        mydata['target'] = torch.from_numpy(target_np).type(torch.float)
        mydata['net_input'] = torch.from_numpy(net_input_np).type(torch.float)

        return mydata

def display_data(target, output, net_input, target_name, net_input_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
    net_input = net_input.cpu().detach()
    if target_name == 'log_full':
        output_np = output[0, 0, :, :]
        target_np = target[0, 0, :, :]
    elif target_name == 'full':
        output_real = output[0, 0, :, :]
        output_imag = output[0, 1, :, :]
        output_mag = np.abs(output_real + 1j * output_imag) + 10 ** (-12)
        output_np = 20 * np.log(output_mag) 
        target_real = target[0, 0, :, :]
        target_imag = target[0, 1, :, :]
        target_mag = np.abs(target_real + 1j * target_imag) + 10 **(-12)
        target_np = 20 * np.log(target_mag)
    elif target_name == 'polar_full':
        # print('output shape:', output.shape)
        output_real = output[0, :, 0: output.shape[2]// 2]
        output_imag = output[0, :, output.shape[2]//2 : output.shape[2]]
        output_mag = np.abs(output_real + 1j * output_imag) + 10 ** (-12)
        output_np = 20 * np.log(output_mag)
        # print('output_imag shape: ', output_imag.shape)
        target_real = target[0, :, 0: target.shape[2]// 2]
        target_imag = target[0, :, target.shape[2]//2 : target.shape[2]]
        target_mag = np.abs(target_real + 1j * target_imag) + 10 ** (-12)
        target_np = 20 * np.log(target_mag)
    else:
        raise NotImplementedError
    
    if net_input_name == 'log_partial':
        input0 = net_input[0, 0, :, :]
        input1 = net_input[0, 1, :, :]
    elif net_input_name == 'partial':
        real0 = net_input[0, 0, :, :]
        imag0 = net_input[0, 1, :, :]
        mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
        input0 = 20 * np.log(mag0)
        real1 = net_input[0, 2, :, :]
        imag1 = net_input[0, 3, :, :]
        mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
        input1 = 20 * np.log(mag1)
    elif net_input_name == 'polar_partial':
        net_input0 = net_input[0, :, 0: net_input.shape[2]//2]
        net_input1 = net_input[0, :, net_input.shape[2]//2 : net_input.shape[2]]
        real0 = net_input0[:, 0: net_input0.shape[1]//2]
        imag0 = net_input0[:, net_input0.shape[1]// 2 : net_input0.shape[1]]
        mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
        input0 = 20 * np.log(mag0)
        real1 = net_input1[:, 0: net_input1.shape[1]//2]
        imag1 = net_input1[:, net_input1.shape[1]// 2 : net_input1.shape[1]]
        mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
        input1 = 20 * np.log(mag1)
    else:
        raise NotImplementedError
    
    extent = [-datagen.max_rng, datagen.max_rng, 0, datagen.max_rng]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(target_np,  extent = extent, origin= 'lower')
    axs[0, 0].set_title('target')
    axs[1, 0].imshow(output_np, extent = extent, origin= 'lower')
    axs[1, 0].set_title('net output')
    axs[0, 1].imshow(input0, extent = extent, origin= 'lower')
    axs[0, 1].set_title('net input 0')
    axs[1, 1].imshow(input1, extent = extent, origin= 'lower')
    axs[1,1].set_title('net input 1')

    plt.show()


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    data_dir = os.path.join('cloud_data', 'points', 'train')

    data_list = os.listdir(data_dir)
    data_path = os.path.join(data_dir, data_list[0])
    # print(data_path)
    # data_info = np.load(data_path)
    # x_points = data_info['all_point_x']
    # print(x_points)
    # y_points = data_info['all_point_y']
    # print(y_points)

    # raw_data = datagen.get_scene_raw_data(x_points, y_points)
    
    # processed_data = datagen.get_radar_image_pairs(raw_data)

    # target_np = processed_data['polar_full']
    # partial0_np = processed_data['polar_partial0']

    # plt.figure()
    # plt.imshow(target_np)
    # plt.title('target')
    # plt.figure()
    # plt.imshow(partial0_np)
    # plt.title('partial0')

    # plt.show()

    mydataset = PointDataSet(data_dir, data_list)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= True, num_workers= 4)
    net_input_name = 'polar_partial'
    target_name = 'polar_full'
    for batch_idx, sample in enumerate(mydataloader):
        for name in sample:
            print(name)
            thing = sample[name]
            print(thing.size())

        target = sample[target_name]
        net_input = sample[net_input_name]
        print(sample['x_points'])
        print(sample['y_points'])
        display_data(target, target, net_input, target_name, net_input_name)

        break