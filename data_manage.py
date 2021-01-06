import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.utils.data as data

import datagen

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
        raw_data = datagen.get_scene_raw_data(x_points, y_points)

        processed = datagen.get_radar_image_pairs(raw_data)

        target_np = processed['log_full']
        partial0_np = processed['log_partial0']
        partial1_np = processed['log_partial1']

        all_data = np.dstack((target_np, partial0_np, partial1_np))
        # print('all data shape: ', all_data.shape)
        # total_min = np.min(np.min(np.min(all_data)))
        # norm_factor = np.max(np.max(np.max(all_data))) - total_min

        # target_np = (target_np - total_min) / norm_factor
        target_np = target_np[np.newaxis, :, :]
        target_torch = torch.from_numpy(target_np).type(torch.float)
        mydata['log_full'] = target_torch
        
        partial_np = all_data[:, :, 1:3] #- total_min) / norm_factor
        # target_np = 
        partial_np = partial_np.transpose(2, 0, 1)
        partial_torch = torch.from_numpy(partial_np).type(torch.float)
        mydata['log_partial'] = partial_torch

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_dir = os.path.join('cloud_data', 'points', 'train')

    data_list = os.listdir(data_dir)
    data_path = os.path.join(data_dir, data_list[0])
    print(data_path)
    data_info = np.load(data_path)
    x_points = data_info['all_point_x']
    print(x_points)
    y_points = data_info['all_point_y']
    print(y_points)

    raw_data = datagen.get_scene_raw_data(x_points, y_points)
    
    processed_data = datagen.get_radar_image_pairs(raw_data)

    target_np = processed_data['log_full']
    partial0_np = processed_data['log_partial0']

    plt.figure()
    plt.imshow(target_np)
    plt.title('target')
    plt.figure()
    plt.imshow(partial0_np)
    plt.title('partial0')

    plt.show()

    mydataset = PointDataSet(data_dir, data_list)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= True, num_workers= 4)

    for batch_idx, sample in enumerate(mydataloader):
        for name in sample:
            print(name)
            thing = sample[name]
            print(thing.size())

        break