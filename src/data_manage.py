import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
import matplotlib.pyplot as plt 
import datagen
import image_transformation_utils as trans 
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class PointDataSet(data.Dataset):
    def __init__(self, data_dir, data_list, net_input_name = 'partial', target_name = 'full', pre_processed = False):
        self.data_dir = data_dir
        self.data_list = data_list
        self.net_input_name = net_input_name
        self.target_name = target_name
        self.pre_processed = pre_processed
    def __len__(self):
        
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        mydata = {}
        filepath = os.path.join(self.data_dir, self.data_list[idx])
        mydata['file_path'] = filepath
        npzfile = np.load(filepath)
        x_points = npzfile['all_point_x']
        # print(x_points)
        y_points = npzfile['all_point_y']
        mydata['x_points'] = x_points
        mydata['y_points'] = y_points

        if self.pre_processed:
            processed = npzfile
        else:
            raw_data = datagen.get_scene_raw_data(x_points, y_points)
            processed = datagen.get_radar_image_pairs(raw_data)

        if self.net_input_name == 'log_partial' or self.net_input_name == 'log_partial_q1':
            partial0_np = processed['mag_partial0']
            partial1_np = processed['mag_partial1']
            partial_np = np.dstack((partial0_np, partial1_np))
            partial_np = 20 * np.log10(partial_np + 10 ** (-12))
            partial_np = partial_np.transpose(2, 0, 1)
            if self.net_input_name == 'log_partial_q1':
                partial_np = partial_np > -20 
            partial_torch = torch.from_numpy(partial_np).type(torch.float)
            mydata[self.net_input_name] = partial_torch
        
        elif self.net_input_name == 'polar_partial':
            polar_partial0_np = processed['polar_partial0']
            polar_partial1_np = processed['polar_partial1']
            # polar_partial_np0 = np.hstack([polar_partial0_np.real, polar_partial0_np.imag])
            # polar_partial_np1 = np.hstack([polar_partial1_np.real, polar_partial1_np.imag])
            polar_partial_np = np.block([[polar_partial0_np.real, polar_partial0_np.imag], [polar_partial1_np.real, polar_partial1_np.imag]])
            polar_partial_torch = torch.from_numpy(polar_partial_np).type(torch.float)
            mydata['polar_partial'] = polar_partial_torch

        elif self.net_input_name == 'polar_partial2d':
            polar_partial0_np = processed['polar_partial0']
            polar_partial1_np = processed['polar_partial1']
            # polar_partial_np0 = np.hstack([polar_partial0_np.real, polar_partial0_np.imag])
            # polar_partial_np1 = np.hstack([polar_partial1_np.real, polar_partial1_np.imag])
            polar_partial_np = np.dstack([polar_partial0_np.real, polar_partial0_np.imag, polar_partial1_np.real, polar_partial1_np.imag])
            polar_partial_np = polar_partial_np.transpose(2, 0, 1)
            polar_partial_torch = torch.from_numpy(polar_partial_np).type(torch.float)
            mydata['polar_partial2d'] = polar_partial_torch
        
        elif self.net_input_name == 'polar_partial2d_q1':
            polar_partial0_np = processed['polar_partial0']
            polar_partial1_np = processed['polar_partial1']
            polar_partial_np = np.dstack([np.abs(polar_partial0_np), np.abs(polar_partial1_np)])
            polar_partial_np = polar_partial_np.transpose(2, 0, 1)
            polar_partial_np = polar_partial_np > 0.5
            polar_partial_torch = torch.from_numpy(polar_partial_np).type(torch.float)
            mydata[self.net_input_name] = polar_partial_torch

        elif self.net_input_name == 'partial':
            real_partial0_np = processed['real_partial0']
            imag_partial0_np = processed['imag_partial0']
            real_partial1_np = processed['real_partial1']
            imag_partial1_np = processed['imag_partial1']
            partial_np = np.dstack((real_partial0_np, imag_partial0_np, real_partial1_np, imag_partial1_np))
            partial_np = partial_np.transpose(2, 0, 1)         
            partial_torch = torch.from_numpy(partial_np).type(torch.float)
            mydata['partial'] = partial_torch
        elif self.net_input_name == 'raw_partial':
            raw0 = raw_data[:, 0: datagen.num_samples]
            raw1 = raw_data[:, raw_data.shape[1] - datagen.num_samples: raw_data.shape[1]]
            raw_np = np.hstack([raw0.real, raw0.imag, raw1.real, raw1.imag])
            # raw_np = raw_np.transpose(2, 0, 1)
            raw_torch = torch.from_numpy(raw_np).type(torch.float)
            mydata[self.net_input_name] = raw_torch
        else:
            raise NotImplementedError('input name not implemented for dataset')

        if self.target_name == 'log_full' or self.target_name == 'log_q1':
            target_np = 20 * np.log10(processed['mag_full'] + 10 **(-12))
            target_np = target_np[np.newaxis, :, :]
            
            if self.target_name == 'log_q1':
                target_np = target_np > 0

            target_torch = torch.from_numpy(target_np).type(torch.float)
            mydata[self.target_name] = target_torch
            
        elif self.target_name == 'polar_full':
            polar_full_np = processed['polar_full']
            polar_full_np = np.hstack([polar_full_np.real, polar_full_np.imag])
            polar_full_torch = torch.from_numpy(polar_full_np).type(torch.float)
            mydata[self.target_name] = polar_full_torch
        elif self.target_name == 'polar_full2d':
            polar_full_np = processed['polar_full']
            polar_full_np = np.dstack([polar_full_np.real, polar_full_np.imag])
            polar_full_np = polar_full_np.transpose(2, 0, 1)
            polar_full_torch = torch.from_numpy(polar_full_np).type(torch.float)
            mydata[self.target_name] = polar_full_torch
        elif self.target_name == 'polar_full2d_q1':
            polar_full_np = processed['polar_full']
            polar_full_np = np.abs(polar_full_np)
            polar_full_np = polar_full_np[np.newaxis, :, :]
            polar_full_np = polar_full_np > 1
            polar_full_torch = torch.from_numpy(polar_full_np).type(torch.float)
            mydata[self.target_name] = polar_full_torch
        elif self.target_name == 'full':
            real_full_np = processed['real_full']
            imag_full_np = processed['imag_full']
            full_np = np.dstack((real_full_np, imag_full_np))
            full_np = full_np.transpose(2, 0, 1)
            full_torch = torch.from_numpy(full_np).type(torch.float)
            mydata[self.target_name] = full_torch
        # elif self.target_name == 'raw_full':
        #     raw_np = np.dstack([raw_data.real, raw_data.imag])
            
        else:
            raise NotImplementedError('target name not implemented for dataset')
        
        return mydata

# class ImageDataSet(data.Dataset):
#     pass

def display_data(target, output, net_input, target_name, net_input_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
    net_input = net_input.cpu().detach()
    if target_name == 'log_full' or target_name == 'log_q1':
        output_np = output[0, 0, :, :]
        target_np = target[0, 0, :, :]
    elif target_name == 'full' or 'polar_full2d':
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
    
    if net_input_name == 'log_partial' or net_input_name == 'log_partial_q1':
        input0 = net_input[0, 0, :, :]
        input1 = net_input[0, 1, :, :]
    elif net_input_name == 'partial' or 'polar_partial2d':
        real0 = net_input[0, 0, :, :]
        imag0 = net_input[0, 1, :, :]
        mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
        input0 = 20 * np.log(mag0)
        real1 = net_input[0, 2, :, :]
        imag1 = net_input[0, 3, :, :]
        mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
        input1 = 20 * np.log(mag1)
    elif net_input_name == 'polar_partial':
        net_input0 = net_input[0, 0:net_input.shape[1]//2, :]
        net_input1 = net_input[0, net_input.shape[1]//2:net_input.shape[1], :]
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

    # img_grid = torchvision.utils.make_grid(images)
    # target_tensor = transforms.ToTensor()(target_np).unsqueeze(0).unsqueeze(0)
    # output_tensor = transforms.ToTensor()(output_np).unsqueeze(0).unsqueeze(0)
    # input0_tensor = transforms.ToTensor()(input0).unsqueeze(0).unsqueeze(0)
    # input1_tensor = transforms.ToTensor()(input1).unsqueeze(0).unsqueeze(0)
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

def get_input_image_grid(net_input, net_input_name):
    net_input = net_input.cpu().detach()
    if net_input_name == 'log_partial' or net_input_name == 'log_partial_q1' or net_input_name == 'polar_partial2d_q1':
        input0 = net_input[0, 0, :, :]
        input1 = net_input[0, 1, :, :]
    elif net_input_name == 'partial' or net_input_name == 'polar_partial2d':
        real0 = net_input[0, 0, :, :]
        imag0 = net_input[0, 1, :, :]
        mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
        input0 = 20 * np.log(mag0)
        real1 = net_input[0, 2, :, :]
        imag1 = net_input[0, 3, :, :]
        mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
        input1 = 20 * np.log(mag1)
    elif net_input_name == 'polar_partial':
        net_input0 = net_input[0, 0:net_input.shape[1]//2, :]
        net_input1 = net_input[0, net_input.shape[1]//2:net_input.shape[1], :]
        real0 = net_input0[:, 0: net_input0.shape[1]//2]
        imag0 = net_input0[:, net_input0.shape[1]// 2 : net_input0.shape[1]]
        mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
        input0 = 20 * np.log(mag0)
        real1 = net_input1[:, 0: net_input1.shape[1]//2]
        imag1 = net_input1[:, net_input1.shape[1]// 2 : net_input1.shape[1]]
        mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
        input1 = 20 * np.log(mag1)
    else:
        raise NotImplementedError('invalid input name')
    input0_tensor = input0.unsqueeze(0)
    input1_tensor = input1.unsqueeze(0)
    inputgrid = torchvision.utils.make_grid([input0_tensor, input1_tensor], padding = 20, pad_value = 1)

    return inputgrid

def get_output_target_image_grid(output, target, target_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
    if target_name == 'log_full' or target_name == 'log_q1' or target_name == 'polar_full2d_q1':
        output_np = output[0, 0, :, :]
        target_np = target[0, 0, :, :]
    elif target_name == 'full' or target_name == 'polar_full2d':
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
        raise NotImplementedError('invalid target name')
    
    output_tensor = output_np.unsqueeze(0)
    target_tensor = target_np.unsqueeze(0)
    img_grid = torchvision.utils.make_grid([output_tensor, target_tensor], padding = 20, pad_value = 1)
    return img_grid

def matplotlib_imshow(img, title = 'input'):
    img_np = img.numpy()
    plt.imshow(img_np[0, :, :], cmap = 'gray')
    plt.title(title)

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    data_dir = os.path.join('../cloud_data', 'vlines', 'val')
    net_input_name = 'partial' #'polar_partial2d'
    target_name = 'log_q1' #'polar_full2d'
    data_list = os.listdir(data_dir)
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
    nums_examine = 1
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')

    mydataset = PointDataSet(data_dir, data_list, net_input_name, target_name, pre_processed = True)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    
    for batch_idx, sample in enumerate(mydataloader):
        for name in sample:
            print(name)
            thing = sample[name]
            if name == net_input_name or name == target_name:
                # print(name)
                print('size: ', thing.size())
                print('minimum value: ', torch.min(thing))
                print('maximum value: ', torch.max(thing))
                print('average value: ', torch.mean(thing))
                print('variance: ', torch.var(thing))
                zero_tensor = torch.zeros(thing.size())
                # print('0:', torch.sum(zero_tensor))
                mynorm = mse(thing, zero_tensor)
                print('sum abs squared: ', mynorm.item())
            else:
                print(thing)
        nums_examined += 1
        print(nums_examined)
        if show_figs:
            target = sample[target_name]
            net_input = sample[net_input_name]
            # print(sample['x_points'])
            # print(sample['y_points'])
            # display_data(target, target, net_input, target_name, net_input_name)
            # plt.figure()
            inputgrid = get_input_image_grid(net_input, net_input_name)
            matplotlib_imshow(inputgrid, 'input')
            plt.figure()
            img_grid = get_output_target_image_grid(target, target, target_name)
            matplotlib_imshow(img_grid, 'target')
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

    #######################################################
    # filepath = os.path.join(data_dir, data_list[0])
    # # mydata['file_path'] = filepath
    # npzfile = np.load(filepath)
    # x_points = npzfile['all_point_x']
    # # print(x_points)
    # y_points = npzfile['all_point_y']
    # # mydata['x_points'] = x_points
    # # mydata['y_points'] = y_points
    # raw_data = datagen.get_scene_raw_data(x_points, y_points)

    # processed = datagen.get_radar_image_pairs(raw_data)
    # polar_partial0_np = processed['polar_partial0']
    # polar_partial1_np = processed['polar_partial1']

    # plt.figure()
    # plt.imshow(np.abs(polar_partial0_np))

    # net_input = np.block([[polar_partial0_np.real, polar_partial0_np.imag], [polar_partial1_np.real, polar_partial1_np.imag]])
    # print(net_input.shape)
    # # net_input0 = net_input[0:2, :]
    # # net_input1 = net_input[2:4, :]
    # # real0 = net_input0[:, 0: net_input0.shape[1]//2]
    # # imag0 = net_input0[:, net_input0.shape[1]// 2 : net_input0.shape[1]]
    # # mag0 = np.abs(real0 + 1j * imag0) + 10 ** (-12)
    # # input0 = 20 * np.log(mag0)
    # # real1 = net_input1[:, 0: net_input1.shape[1]//2]
    # # imag1 = net_input1[:, net_input1.shape[1]// 2 : net_input1.shape[1]]
    # # mag1 = np.abs(real1 + 1j * imag1) + 10 **(-12)
    # # input1 = 20 * np.log(mag1)

    # # plt.figure()
    # # plt.imshow(input1)
    # # plt.title('input 1')
    # plt.show()