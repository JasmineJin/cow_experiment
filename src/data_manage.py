import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
import matplotlib.pyplot as plt 
import datagen_new as gen
import image_transformation_utils as trans 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
#####################################################################################################################
#
# we are making strides here! 
#
# our new version PointDataSet is going to assume the input data has phase info like in notes from 3/26
#
#####################################################################################################################
class PointDataSet(data.Dataset):
    def __init__(self, data_dir, data_list, pre_processed = False):
        self.data_dir = data_dir
        self.data_list = data_list
        self.net_input_name = net_input_name
        self.target_name = target_name
        self.pre_processed = pre_processed
        # self.things_to_include = things_to_include
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

        mydata['polar_full'] = torch.from_numpy(npzfile['polar_full']).type(torch.float)
        mydata['polar_partial_mag_phase'] = torch.from_numpy(npzfile['polar_partial_mag_phase']).type(torch.float)
        
        return mydata

def norm01(data):
    min_data = torch.min(data)
    data = data - min_data
    data = data / torch.max(data)
    return data

def quantizer(data, low, high, n_levels):
    """
    high = low + (n_levels - 1) * delta
    delta = (high - low) / (n_levels - 1)
    input: numpy array
    output: integers in range[0, n_levels - 1]
    """
    delta = (high - low) / (n_levels)
    data = data.clone()
    data[data < low] = low
    data[data > low + (n_levels - 1) * delta] = low + (n_levels - 1) * delta
    
    
    data = np.floor((data - low) / delta)

    return data

def dequantizer(quantized_data, low, high, n_levels):
    return quantized_data


def get_input_image_grid(net_input, net_input_name):
    net_input = net_input.cpu().detach()
    # if net_input_name == 'polar_partial_mag_phase':
    real0 = net_input[0, 0, :, :]
    imag0 = net_input[0, 1, :, :]
    mag0 = np.abs(real0 + 1j * imag0)
    input0 = mag0
    real1 = net_input[0, 2, :, :]
    imag1 = net_input[0, 3, :, :]
    mag1 = np.abs(real1 + 1j * imag1)
    input1 = mag1

    input0_tensor = input0.unsqueeze(0)
    input1_tensor = input1.unsqueeze(0)
    inputgrid = torchvision.utils.make_grid([input0_tensor, input1_tensor], padding = 20, pad_value = 1)

    return inputgrid

def get_output_target_image_grid(output, target, target_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
    output_np = output[0, 0, :, :]
    target_np = target[0, 0, :, :]
    
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
    data_dir = os.path.join('cloud_data', 'moooo', 'val')
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)
    print(len(data_list))
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
    nums_examine = 2
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')

    mydataset = PointDataSet(data_dir, data_list)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    
    myiter = iter(mydataloader)
    for i in range(nums_examine):
        sample = myiter.next()

        # print(thing)

    # for batch_idx, sample in enumerate(mydataloader):
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
            target = norm01(target)
            net_input = sample[net_input_name]
            net_input = norm01(net_input)
            # print(sample['x_points'])
            # print(sample['y_points'])
            # display_data(target, target, net_input, target_name, net_input_name)
            plt.figure()
            inputgrid = get_input_image_grid(net_input, net_input_name)
            matplotlib_imshow(inputgrid, 'input')
            plt.figure()
            img_grid = get_output_target_image_grid(target, target, target_name)
            matplotlib_imshow(img_grid, 'target')
            # print(inputgrid.size())
            plt.show()
        
        # if nums_examined >= nums_examine:
        #     break

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