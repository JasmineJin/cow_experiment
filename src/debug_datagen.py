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


if __name__ == '__main__':
    
    data_dir = os.path.join('cloud_data', 'vline', 'debug')
    # net_input_name = 'polar_partial2d_q1'
    # target_name = 'log_full'
    data_list = os.listdir(data_dir)
    idx = 0
    filepath = os.path.join(data_dir, data_list[idx])

    processed_saved = np.load(filepath)

    x_points = processed_saved['all_point_x']
        # print(x_points)
    y_points = processed_saved['all_point_y']
    raw_data = gen.get_scene_raw_data(x_points, y_points)
    processed_new = gen.get_radar_image_pairs(raw_data)

    raw_data_saved = processed_saved['raw_data']
    
    raw_data_diff = np.abs(raw_data - raw_data_saved)
    print('raw_data changed by: ', np.sum(np.sum(raw_data_diff)))

    raw_data0 = raw_data[:, 0: gen.num_samples]
    raw_data0_saved = raw_data_saved[:, 0: gen.num_samples]#
    raw_data0_diff = np.abs(raw_data0 - raw_data0_saved)
    print('raw_data0 changed by: ', np.mean(np.mean(raw_data0_diff)))
    print('raw data new l1 norm: ', np.sum(np.sum(np.abs(raw_data0))))
    print('raw data saved l1 norm: ', np.sum(np.sum(np.abs(raw_data0_saved))))
    print('raw data saved shape: ', raw_data0_saved.shape)

    polar0 = gen.process_array(raw_data0)
    # threshold_mtx = gen.apply_threshold_per_row(20 * np.log10(polar0), 25, 25)
    # polar0 = polar0 #* threshold_mtx

    polar0_saved = processed_saved['polar_partial0']
    polar0_diff = np.abs(polar0 - polar0_saved)
    print('diff in polar0: ', np.mean(np.mean(polar0_diff)))


    # mag_full_saved = processed_saved['mag_full']

    # # mag_full_new = processed_new['mag_full']

    # diff = mag_full_saved - mag_full_new

    # # print(diff.shape)
    # print('diff in mag_full', np.mean(np.mean(np.abs(diff))))


    plt.figure()
    plt.imshow(np.abs(polar0))
    plt.title('new version')

    plt.figure()
    plt.imshow(np.abs(polar0_saved))
    plt.title('saved version')
    plt.show()