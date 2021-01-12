import numpy as np
# import matplotlib

import numpy as np
import scipy as sp
from numpy.fft import fft, ifft, fftshift, ifftshift
import os
import scipy as sp
import image_transformation_utils as trans
##############################################################################################
# global constants
num_range_bins = 256
num_channels = 512
wl= (299.792458 / 77) * 0.001
num_samples = 16
center_freq = 77 * 10 ** 9
speed_of_light = 299792458
bandwidth = 2 * 10 ** 9
max_velocity = 20

normal_h = 256
normal_w = 512

cycle_length = wl /  (4 * max_velocity) # Vmax = lambda/(4 * Tc)
rng_res = speed_of_light / (2 * bandwidth)
cos_angle_res = 2 / num_channels
max_rng = num_range_bins * rng_res
slope = bandwidth / cycle_length
antenna_spacing = wl / 2
antenna_array = np.arange(num_channels) * antenna_spacing
antenna_array = antenna_array - antenna_array[-1] / 2

Pmax = 100
################################################################################################
##### data generation code 

def add_point(point_x, point_y):

    """
    return the raw data of radar response for single point source at [point_x, point_y]
    Pmax = max power
    plot figures if show_figs
    """
    x_diff = antenna_array - point_x
    y_diff = point_y

    obj_rng = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    n_vector = np.arange(num_range_bins)
    n_vector = n_vector[:, np.newaxis]

    phase0 = -2 * np.pi * obj_rng[0] / wl
    phase_diff = -2 * np.pi * (obj_rng - obj_rng[0]) / wl
    phase = np.tile(phase0 + phase_diff, (num_range_bins, 1))

    array_response = np.exp(1j *(( 2 * np.pi * n_vector / num_range_bins) * (obj_rng * 2 * bandwidth/ speed_of_light) + phase))

    power_scaling = Pmax / (obj_rng * obj_rng)
    power_scaling_mtx = np.diag(power_scaling)

    array_response = np.dot(array_response, power_scaling_mtx)

    return array_response

def process_array(array_response):
    """
    process array
    """
    # print('array response shape: ', array_response.shape)
    range_window = np.hanning(num_range_bins)
    range_window_mtx = np.diag(range_window)

    num_channels_to_process = array_response.shape[1]
    channel_window = np.hanning(num_channels_to_process)
    channel_window_mtx = np.diag(channel_window)

    range_processed = fft(np.dot(range_window_mtx, array_response), axis = 0, norm= 'ortho')

    angle_processed = fft(np.dot(range_processed, channel_window_mtx), n = num_channels, axis= 1, norm = 'ortho')
    for r in range(num_range_bins):
        angle_processed[r, :] = np.roll(angle_processed[r, :], num_channels // 2)

    return angle_processed

def apply_threshold_per_row(matrix, dip1, dip_oeverall):
    max_value = np.max(np.max(matrix))
    threshold_overall = max_value - dip_oeverall
    thresholding_mtx = np.ones(matrix.shape)
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        peak = np.max(row)
        thresholding_mtx[i, row < (peak - dip1)] = 0
        thresholding_mtx[i, row < threshold_overall] = 0
    return thresholding_mtx
        
def get_scene_raw_data(all_point_x, all_point_y, save_path = ''):
    """
    given collection of points whose x coordinates are in all_point_x and y coordinates are in all_point_y,
    save the raw data to save_path if save path is not empty
    """
    assert(len(all_point_x) == len(all_point_y))
    num_points = len(all_point_x)

    radar_response = np.zeros((num_range_bins, num_channels), dtype = np.complex128)    
    for i in range(num_points):
 
        point_x = all_point_x[i]
        point_y = all_point_y[i]
        radar_response += add_point(point_x, point_y)

    if save_path != '':
        np.savez_compressed(save_path, 
                    raw_data = radar_response,
                    x_points = all_point_x,
                    y_points = all_point_y)

    return radar_response

def get_normal_plot_label(all_point_x, all_point_y):
    """
    given the list of points, create the ground truth plot in xy coordinates
    """
    assert(len(all_point_x) == len(all_point_y))
    num_points = len(all_point_x)

    normal_plot_label = np.zeros((normal_h, normal_w))
    rng_vector = np.arange(num_range_bins) * rng_res
    _, x, y = trans.polar_to_rect(normal_plot_label, wl, num_channels, rng_vector, normal_h, normal_w)
    
    x_res = x[1] - x[0]
    y_res = y[1] - y[0]
    for i in range(num_points):
        point_x = all_point_x[i]
        point_y = all_point_y[i]
        col = int((point_x - x[0])/ x_res)    
        row = int((point_y - y[0])/ y_res)
        normal_plot_label[row, col] = 1

    return normal_plot_label

def get_radar_image_pairs(radar_response):
    """
    given raw data, return pairs of high resolution and low resolution image
    """
    radar_ra_plot = process_array(radar_response)
    rng_vector = np.arange(num_range_bins) * rng_res
    thresholding_mtx = apply_threshold_per_row(20 * np.log10(np.abs(radar_ra_plot)), 25, 25)

    radar_ra_plot_thresholded = radar_ra_plot * thresholding_mtx
    normal_plot_real, x, y = trans.polar_to_rect(np.real(radar_ra_plot_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot_imag, x, y = trans.polar_to_rect(np.imag(radar_ra_plot_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot = np.abs(normal_plot_real + 1j * normal_plot_imag)
    # normal_plot = np.log10(normal_plot + 10** (-12))

    radar_ra_plot_partial0 = process_array(radar_response[:, 0: num_samples])
    thresholding_mtx = apply_threshold_per_row(20 * np.log10(np.abs(radar_ra_plot_partial0)), 25, 25)
    radar_ra_plot_partial0_thresholded = radar_ra_plot_partial0 * thresholding_mtx

    normal_plot_partial0_real, x, y = trans.polar_to_rect(np.real(radar_ra_plot_partial0_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot_partial0_imag, x, y = trans.polar_to_rect(np.imag(radar_ra_plot_partial0_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot_partial0 = np.abs(normal_plot_partial0_real + 1j * normal_plot_partial0_imag)
    # normal_plot_partial0 = np.log10(normal_plot_partial0 + 10 ** (-12))

    radar_ra_plot_partial1 = process_array(radar_response[:, num_channels - num_samples : num_channels])
    thresholding_mtx = apply_threshold_per_row(20 * np.log10(np.abs(radar_ra_plot_partial1)), 25, 25)
    radar_ra_plot_partial1_thresholded = radar_ra_plot_partial1 * thresholding_mtx

    normal_plot_partial1_real, x, y = trans.polar_to_rect(np.real(radar_ra_plot_partial1_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot_partial1_imag, x, y = trans.polar_to_rect(np.imag(radar_ra_plot_partial1_thresholded), wl, num_channels, rng_vector, 256, 512)
    normal_plot_partial1 = np.abs(normal_plot_partial1_real + 1j * normal_plot_partial1_imag)
    # normal_plot_partial1 = np.log10(normal_plot_partial1 + 10 ** (-12))

    output = {'mag_full': normal_plot, 
                'real_full': normal_plot_real,
                'imag_full': normal_plot_imag,
                'mag_partial0': normal_plot_partial0, 
                'real_partial0': normal_plot_partial0_real,
                'imag_partial0': normal_plot_partial0_imag,
                'mag_partial1': normal_plot_partial1, 
                'real_partial1': normal_plot_partial1_real,
                'imag_partial1': normal_plot_partial1_imag,
                'polar_full': radar_ra_plot_thresholded,
                'polar_partial0': radar_ra_plot_partial0_thresholded,
                'polar_partial1': radar_ra_plot_partial1_thresholded}
    
    return output

    # plt.figure()
    # plt.imshow(np.abs(radar_ra_plot), extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle plot full array')

    

    # plt.figure()
    # plt.imshow(normal_plot, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('full array image')

    # plt.figure()
    # plt.imshow(np.abs(radar_ra_plot_partial0), extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle plot first 16 antennas')


    # plt.figure()
    # plt.imshow(normal_plot_partial0, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('first 16 antennas image')

    # plt.figure()
    # plt.imshow(range_angle_label, extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle locations for points')


    # plt.figure()
    # plt.imshow(normal_plot_label, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('xy locations for points')

    # plt.show()

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='set mode for data generation')
    parser.add_argument('--mode', type = str, default = 'train')
    parser.add_argument('--max_num_points', type = int, default = 1)
    parser.add_argument('--num_samples', type = int, default = 1000)
    args = parser.parse_args()

    mode = args.mode
    max_num_points = args.max_num_points
    num_samples = args.num_samples
    print(mode)
    print(max_num_points)
    print(num_samples)

    
    # if mode == 'val':
    # mode = 'val'
    # max_num_points = 1
    # num_samples = 1000
    data_dir = os.path.join('cloud_data', 'points', mode)
    os.makedirs(data_dir, exist_ok = True)
    for num_points in range(1, max_num_points + 1):
        print('making scenes with ', num_points, ' point sources')
        for n in range(num_samples):
            all_ranges = np.random.rand(num_points) * max_rng
            all_angles = np.random.rand(num_points) * np.pi
            all_point_x = all_ranges * np.cos(all_angles)
            all_point_y = all_ranges * np.sin(all_angles)
            scene_name = 'point_' + str(num_points) +'source_' + str(n) + '.npz'
            save_path = os.path.join(data_dir, scene_name)
            np.savez_compressed(save_path, all_point_x = all_point_x, all_point_y = all_point_y)
            # get_scene_raw_data(all_point_x, all_point_y, save_path)
            
            if n % 100 == 0:
                print(n)

    # range_angle_label = np.zeros((num_range_bins, num_channels))
    # radar_response = np.zeros((num_range_bins, num_channels), dtype = np.complex128)
    # rng_vector = np.arange(num_range_bins) * rng_res
    # _, x, y = trans.polar_to_rect(range_angle_label, wl, num_channels, rng_vector, 256, 512)
    # normal_plot_label = np.zeros((256, 512))
    # x_res = x[1] - x[0]
    # y_res = y[1] - y[0]

    # for i in range(num_points):
 
    #     point_x = all_point_x[i]
    #     point_y = all_point_y[i]
    #     radar_response += add_point(point_x, point_y)

    #     # cos_angle =
    #     r = np.sqrt(point_x **2 + point_y ** 2)
    #     cos_angle = point_x / r + 1
        
    #     column_number = int(np.floor(cos_angle / cos_angle_res))
    #     row_number = int(np.floor(r / rng_res))
    #     range_angle_label[row_number, column_number] = 1

    #     col = int((point_x - x[0])/ x_res)    
    #     row = int((point_y - y[0])/ y_res)
    #     normal_plot_label[row, col] = 1

    # radar_ra_plot = process_array(radar_response)
    # rng_vector = np.arange(num_range_bins) * rng_res
    # thresholding_mtx = apply_threshold_per_row(20 * np.log10(np.abs(radar_ra_plot)), 25, 50)

    # radar_ra_plot_thresholded = radar_ra_plot * thresholding_mtx
    # normal_plot_real, x, y = trans.polar_to_rect(np.real(radar_ra_plot_thresholded), wl, num_channels, rng_vector, 256, 512)
    # normal_plot_imag, x, y = trans.polar_to_rect(np.imag(radar_ra_plot_thresholded), wl, num_channels, rng_vector, 256, 512)

    # normal_plot = np.abs(normal_plot_real + 1j * normal_plot_imag)

    # plt.figure()
    # plt.imshow(np.abs(radar_ra_plot), extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle plot full array')

    # normal_plot = np.log10(normal_plot + 10** (-12))

    # plt.figure()
    # plt.imshow(normal_plot, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('full array image')

    # radar_ra_plot_partial0 = process_array(radar_response[:, 0: num_samples])
    # thresholding_mtx = apply_threshold_per_row(20 * np.log10(np.abs(radar_ra_plot_partial0)), 25, 50)
    # radar_ra_plot_partial0_thresholded = radar_ra_plot_partial0 * thresholding_mtx

    # normal_plot_partial0_real, x, y = trans.polar_to_rect(np.real(radar_ra_plot_partial0_thresholded), wl, num_channels, rng_vector, 256, 512)
    # normal_plot_partial0_imag, x, y = trans.polar_to_rect(np.imag(radar_ra_plot_partial0_thresholded), wl, num_channels, rng_vector, 256, 512)
    # normal_plot_partial0 = np.abs(normal_plot_partial0_real + 1j * normal_plot_partial0_imag)
    
    # normal_plot_partial0 = np.log10(normal_plot_partial0 + 10 ** (-12))
    # plt.figure()
    # plt.imshow(np.abs(radar_ra_plot_partial0), extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle plot first 16 antennas')


    # plt.figure()
    # plt.imshow(normal_plot_partial0, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('first 16 antennas image')

    # # normal_plot_label, x, y = trans.polar_to_rect(range_angle_label, wl, num_channels, rng_vector, 256, 512)
    # plt.figure()
    # plt.imshow(range_angle_label, extent = [-1, 1, 0, max_rng], aspect = 'auto', origin= 'lower')
    # plt.xlabel('cos(angle)')
    # plt.ylabel('range (m)')
    # plt.title('range angle locations for points')


    # plt.figure()
    # plt.imshow(normal_plot_label, extent = [x[0], x[-1], y[0], y[-1]], origin= 'lower')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.title('xy locations for points')

    # plt.show()



