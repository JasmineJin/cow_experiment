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
# import data_manage as mydata


if __name__ == '__main__':
    extent = [-1, 1, 0, gen.max_rng]

    x_points = np.array([18.8571, -15.3885, 25.7558, -9.0010])
    y_points = np.array([5.8979, 7.5325, 18.4813, 14.1987])
    raw_data = gen.get_scene_raw_data(x_points, y_points)
    # processed_new = gen.get_radar_image_pairs(raw_data)
    processed_0 = gen.get_radar_image_pairs_new(raw_data)

    fig, axs = plt.subplots(2, 2)
    pos0 = axs[0, 0].imshow(processed_0['phase_cos0'], extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[0, 0].set_title('cos(phase) of range angle plot from 12 radar elements')
    axs[0, 0].set_xlabel('cos(AoA)')
    axs[0, 0].set_ylabel('range (m)')
    # cbar = fig.colorbar(pos0)
    pos1 = axs[0, 1].imshow(processed_0['phase_sin0'], extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[0, 1].set_title('sin(phase) of range angle plot from 12 radar elements')
    axs[0, 1].set_xlabel('cos(AoA)')
    axs[0, 1].set_ylabel('range (m)')
    # cbar = fig.colorbar(pos1)
    pos3 = axs[1, 0].imshow(processed_0['multiplied_cos0'], extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[1, 0].set_title('normalized magnitude times cos(phase)')
    axs[1, 0].set_xlabel('cos(AoA)')
    axs[1, 0].set_ylabel('range (m)')
    # cbar = fig.colorbar(pos3)
    pos4 = axs[1, 1].imshow(processed_0['multiplied_sin0'], extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    axs[1, 1].set_title('normalized magnitude times sin(phase)')
    axs[1, 1].set_xlabel('cos(AoA)')
    axs[1, 1].set_ylabel('range (m)')
    # cbar = fig.colorbar(pos4)
    
    plt.figure()
    total = np.abs(processed_0['multiplied_cos0'] + 1j * processed_0['multiplied_sin0'])
    pos = plt.imshow(total, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('Recovered normalized log-magnitude')
    cbar = fig.colorbar(pos)
    
    plt.show()