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
    raw_data = datagen.get_scene_raw_data(x_points, y_points)
    processed_new = datagen.get_radar_image_pairs(raw_data)

    mag_partial0_saved = processed_saved['mag_partial0']

    mag_partial0_new = processed_new['mag_partial0']

    diff = mag_partial0_saved - mag_partial0_new

    print(diff.shape)
    print('total diff', np.sum(np.sum(np.abs(diff))))


    plt.figure()
    plt.imshow(mag_partial0_saved)
    plt.title('saved version')

    plt.figure()
    plt.imshow(mag_partial0_new)
    plt.title('new version')
    plt.show()