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
    processed_new = gen.get_radar_image_pairs(raw_data)
    processed_0 = gen.get_radar_image_pairs_new(raw_data)
    # raw_data0 = raw_data[:, 0: gen.num_samples]
    polar_full = gen.process_array(raw_data)
    polar_full_db = 20 * np.log10(np.abs(polar_full))
    fig = plt.figure()
    pos = plt.imshow(polar_full_db, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('range-angle processing with 1024 radar elements')
    cbar = fig.colorbar(pos, label = 'dB')

    raw_data0 = raw_data[:, 0: gen.num_samples]
    polar0 = gen.process_array(raw_data0)
    polar0_db = 20 * np.log10(np.abs(polar0))
    fig = plt.figure()
    pos = plt.imshow(polar0_db, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('range-angle processing with first 12 radar elements')
    cbar = fig.colorbar(pos, label = 'dB')

    raw_data1 = raw_data[:, 1012: 1024]
    polar0 = gen.process_array(raw_data0)
    polar0_db = 20 * np.log10(np.abs(polar0))
    fig = plt.figure()
    pos = plt.imshow(polar0_db, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('range-angle processing with last 12 radar elements')
    cbar = fig.colorbar(pos)
    # cbar.set_title('dB')

    fig = plt.figure()
    normalized_db_0 = gen.norm01_2d(polar0_db)
    pos = plt.imshow(normalized_db_0, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    plt.xlabel('cos(AoA)')
    plt.ylabel('range (m)')
    plt.title('normalized range-angle estimation with 12 radar elements')
    cbar = fig.colorbar(pos)

    # polar_full_db_thresholded = gen.apply_threshold_per_row(polar_full_db, 20, 50)
    # fig = plt.figure()
    # pos = plt.imshow(polar_full_db_thresholded, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    # plt.xlabel('cos(AoA)')
    # plt.ylabel('range (m)')
    # plt.title(' thresholded range-angle estimation with 1024 radar elements')
    # cbar = fig.colorbar(pos)
    

    # polar0_db_thresholded = gen.apply_threshold_per_row(polar0_db, 20, 50)
    # fig = plt.figure()
    # pos = plt.imshow(polar0_db_thresholded, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    # plt.xlabel('cos(AoA)')
    # plt.ylabel('range (m)')
    # plt.title(' thresholded range-angle estimation with 12 radar elements')
    # cbar = fig.colorbar(pos)

    # fig = plt.figure()
    # normalized_db_full = gen.norm01_2d(polar_full_db)
    # pos = plt.imshow(normalized_db_full, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    # plt.xlabel('cos(AoA)')
    # plt.ylabel('range (m)')
    # plt.title('normalized range-angle estimation with 1024 radar elements')
    # cbar = fig.colorbar(pos)

    # just_phase0 = processed_new['just_phase'][0, :, :]
    # polar0_db_thresholded = gen.apply_threshold_per_row(polar0_db, 20, 50)
    # fig = plt.figure()
    # pos = plt.imshow(just_phase0, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    # plt.xlabel('cos(AoA)')
    # plt.ylabel('range (m)')
    # plt.title('phase of range-angle estimation with 12 radar elements')
    # cbar = fig.colorbar(pos)

    # rng_vector = np.arange(gen.num_range_bins) * gen.rng_res
    # normal_plot, x, y = trans.polar_to_rect(polar_full_db_thresholded, gen.wl, gen.num_channels, rng_vector, 512, 1024)
    # normal_plot = normal_plot > 0
    # fig = plt.figure()
    # new_ext = [x[0], x[-1], y[0], y[-1]]
    # pos = plt.imshow(normal_plot, extent = extent, origin= 'lower', aspect = 'auto', cmap = 'gray')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('Location Estimate from 1024-element array')
    # cbar = fig.colorbar(pos)
    
    
    # print(just_phase.shape)
    plt.show()