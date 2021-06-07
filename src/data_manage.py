import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data
import matplotlib.pyplot as plt 
import data_generation as gen
import torchvision
import torchvision.transforms as transforms
from PIL import Image
#################################################################################################################
# dataset for training with pytorch
# other fun plotting functions to visualize data
#################################################################################################################
class PointDataSet(data.Dataset):
    def __init__(self, data_dir, data_list, pre_processed = False):
        """
        data_dir: the directory where data is stored
        data_list: the list of samples in data_dir to include in the dataset
        pre_processed: True --> directly read from saved data or False --> re-process data from point locations
        """
        self.data_dir = data_dir
        self.data_list = data_list
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
        y_points = npzfile['all_point_y']
        mydata['points_x'] = x_points
        mydata['points_y'] = y_points
        if  self.pre_processed:

            mydata['polar_full'] = torch.from_numpy(npzfile['polar_full']).type(torch.float)
            mydata['polar_partial_mag_phase'] = torch.from_numpy(npzfile['polar_partial_mag_phase']).type(torch.float)
        else:
            raw_data = gen.get_scene_raw_data(x_points, y_points)
            processed = gen.get_radar_image_pairs(raw_data)
            mydata['raw_data'] = raw_data
            mydata['polar_full'] = torch.from_numpy(processed['polar_full']).type(torch.float)
            mydata['polar_partial_mag_phase'] = torch.from_numpy(processed['polar_partial_mag_phase']).type(torch.float)
        return mydata

def norm01(data):
    min_data = torch.min(data)
    data = data - min_data
    data = data / torch.max(data)
    return data

def get_output_target_image_grid(output, target, target_name):
    target = target.cpu().detach()
    output = output.cpu().detach()
    output_np = output[0, 0, :, :]
    target_np = target[0, 0, :, :]
    
    output_tensor = output_np.unsqueeze(0)
    target_tensor = target_np.unsqueeze(0)
    img_grid = torchvision.utils.make_grid([output_tensor, target_tensor], padding = 20, pad_value = 1)
    return img_grid


def plot_just_input_grid(net_input):
    net_input = net_input.cpu().detach()
    real0 = net_input[0, 0, :, :]
    imag0 = net_input[0, 1, :, :]
    mag0 = np.abs(real0 + 1j * imag0)
    input0 = mag0
    real1 = net_input[0, 2, :, :]
    imag1 = net_input[0, 3, :, :]
    mag1 = np.abs(real1 + 1j * imag1)
    input1 = mag1
    extent = [-1, 1, 0, gen.max_rng]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(input0,  extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[0].set_title('Input 0')
    axs[0].set_xlabel('cos(AoA)')
    axs[0].set_ylabel('range (m)')
    axs[1].imshow(input1, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[1].set_title('Input 1')
    axs[1].set_xlabel('cos(AoA)')
    axs[1].set_ylabel('range (m)')


def plot_labeled_grid(net_input, output, target):
    target = target.cpu().detach()
    output = output.cpu().detach()
    output_np = output[0, 0, :, :]
    target_np = target[0, 0, :, :]

    net_input = net_input.cpu().detach()
    real0 = net_input[0, 0, :, :]
    imag0 = net_input[0, 1, :, :]
    mag0 = np.abs(real0 + 1j * imag0)
    input0 = mag0
    real1 = net_input[0, 2, :, :]
    imag1 = net_input[0, 3, :, :]
    mag1 = np.abs(real1 + 1j * imag1)
    input1 = mag1

    extent = [-1, 1, 0, gen.max_rng]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(target_np,  extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[0, 0].set_title('target')
    axs[0, 0].set_xlabel('cos(AoA)')
    axs[0, 0].set_ylabel('range (m)')
    axs[1, 0].imshow(output_np, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[1, 0].set_title('net output')
    axs[1, 0].set_xlabel('cos(AoA)')
    axs[1, 0].set_ylabel('range (m)')
    axs[0, 1].imshow(input0, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[0, 1].set_title('net input 0')
    axs[0, 1].set_xlabel('cos(AoA)')
    axs[0, 1].set_ylabel('range (m)')
    axs[1, 1].imshow(input1, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[1,1].set_title('net input 1')
    axs[1, 1].set_xlabel('cos(AoA)')
    axs[1, 1].set_ylabel('range (m)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse arguments for checking data')
    parser.add_argument('--data_directory', nargs='+', default=['cloud_data', 'mooooo', 'debug'])
    parser.add_argument('--nums_show', type = int, default = 100)
    args = parser.parse_args()
    data_dir = os.path.join(*args.data_directory)

    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)
    print('num_samples in directory: ', len(data_list))
    ###############################################################################
    # check and plot data
    ###############################################################################
    show_figs = True
    check_all = False
    nums_examine = args.nums_show
    nums_examined = 0

    mse = nn.MSELoss(reduction = 'sum')

    mydataset = PointDataSet(data_dir, data_list)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    
    myiter = iter(mydataloader)
    for i in range(nums_examine):
        sample = myiter.next()

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
            plot_just_input_grid(net_input)

            output = torch.ones(target.size())

            plt.figure()
            target = target.cpu().detach()
            output = output.cpu().detach()
            output_np = output[0, 0, :, :]
            target_np = target[0, 0, :, :]
            extent = [-1, 1, 0, gen.max_rng]
            plt.imshow(target_np, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
            plt.xlabel('cos(AoA)')
            plt.ylabel('range (m)')
            plt.show()