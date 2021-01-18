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

class PointDataSet(data.Dataset):
    def __init__(self, data_dir, data_list, net_input_name = 'partial', target_name = 'full'):
        self.data_dir = data_dir
        self.data_list = data_list
        self.net_input_name = net_input_name
        self.target_name = target_name
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
        raw_data = datagen.get_scene_raw_data(x_points, y_points)

        processed = datagen.get_radar_image_pairs(raw_data)

        if self.net_input_name == 'log_partial':
            partial0_np = processed['mag_partial0']
            partial1_np = processed['mag_partial1']
            partial_np = np.dstack((partial0_np, partial1_np))
            partial_np = 20 * np.log10(partial_np + 10 ** (-12))
            partial_np = partial_np.transpose(2, 0, 1)
            partial_torch = torch.from_numpy(partial_np).type(torch.float)
            mydata['log_partial'] = partial_torch
        
        elif self.net_input_name == 'polar_partial':
            polar_partial0_np = processed['polar_partial0']
            polar_partial1_np = processed['polar_partial1']
            # polar_partial_np0 = np.hstack([polar_partial0_np.real, polar_partial0_np.imag])
            # polar_partial_np1 = np.hstack([polar_partial1_np.real, polar_partial1_np.imag])
            polar_partial_np = np.block([[polar_partial0_np.real, polar_partial0_np.imag], [polar_partial1_np.real, polar_partial1_np.imag]])
            polar_partial_torch = torch.from_numpy(polar_partial_np).type(torch.float)
            mydata['polar_partial'] = polar_partial_torch
        elif self.net_input_name == 'partial':
            real_partial0_np = processed['real_partial0']
            imag_partial0_np = processed['imag_partial0']
            real_partial1_np = processed['real_partial1']
            imag_partial1_np = processed['imag_partial1']
            partial_np = np.dstack((real_partial0_np, imag_partial0_np, real_partial1_np, imag_partial1_np))
            partial_np = partial_np.transpose(2, 0, 1)         
            partial_torch = torch.from_numpy(partial_np).type(torch.float)
            mydata['partial'] = partial_torch
        else:
            raise NotImplementedError

        if self.target_name == 'log_full':
            target_np = 20 * np.log10(processed['mag_full'] + 10 **(-12))
            target_np = target_np[np.newaxis, :, :]
            target_torch = torch.from_numpy(target_np).type(torch.float)
            mydata[self.target_name] = target_torch
        elif self.target_name == 'polar_full':
            polar_full_np = processed['polar_full']
            polar_full_np = np.hstack([polar_full_np.real, polar_full_np.imag])
            polar_full_torch = torch.from_numpy(polar_full_np).type(torch.float)
            mydata[self.target_name] = polar_full_torch
        elif self.target_name == 'full':
            real_full_np = processed['real_full']
            imag_full_np = processed['imag_full']
            full_np = np.dstack((real_full_np, imag_full_np))
            full_np = full_np.transpose(2, 0, 1)
            full_torch = torch.from_numpy(full_np).type(torch.float)
            mydata[self.target_name] = full_torch
        else:
            raise NotImplementedError
        
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
    inputgrid = torchvision.utils.make_grid([input0_tensor, input1_tensor], padding = 20)

    return inputgrid

def get_output_target_image_grid(output, target, target_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
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
        raise NotImplementedError('invalid target name')
    
    output_tensor = output_np.unsqueeze(0)
    target_tensor = target_np.unsqueeze(0)
    img_grid = torchvision.utils.make_grid([output_tensor, target_tensor], padding = 20)
    return img_grid

def matplotlib_imshow(img):
    img_np = img.numpy()
    plt.imshow(img_np[0, :, :])

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    data_dir = os.path.join('../cloud_data', 'points', 'train')
    net_input_name = 'polar_partial'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)
    # data_path = os.path.join(data_dir, data_list[0], net_input_name, target_name)

    show_figs = True
    nums_examine = 1
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')

    mydataset = PointDataSet(data_dir, data_list, net_input_name, target_name)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    
    for batch_idx, sample in enumerate(mydataloader):
        for name in sample:
            print(name)
            thing = sample[name]
            if name == net_input_name or name == target_name:
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
            inputgrid = get_input_image_grid(net_input, net_input_name)
            matplotlib_imshow(inputgrid)
            img_grid = get_output_target_image_grid(target, target, target_name)
            matplotlib_imshow(img_grid)
            # print(inputgrid.size())
            plt.show()
        
        if nums_examined >= nums_examine:
            break

        ##################################################################################
        # check the data process
        ##################################################################################

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