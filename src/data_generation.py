############################################################################################
# only saving un-thresholded polar data
############################################################################################
import numpy as np
import scipy as sp
from numpy.fft import fft, ifft, fftshift, ifftshift
import os
import scipy as sp
import copy
##############################################################################################
# global constants
num_range_bins = 512
num_channels = 1024
wl= (299.792458 / 77) * 0.001
num_samples = 12
center_freq = 77 * 10 ** 9
speed_of_light = 299792458
bandwidth = 2 * 10 ** 9
max_velocity = 20

normal_h = 512
normal_w = 1024

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
# data generation code 

def add_point(point_x, point_y):
    """
    Inputs:
        point_x: x coordinate of the ideal point reflector
        point_y: y_coordinate of the ideal point reflector
    output:
        array_response: 2d complex numpy array, the echoed signals from the ideal point reflector
                        at each radar element
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
    input:
        array_response: complex 2d-numpy array, raw echoed signals
    output:
        angle_processed: complex 2d-numpy array 2d-dft results of input with windowing
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

def get_scene_raw_data(all_point_x, all_point_y):
    """
    inputs: 
        all_point_x: list of x-coordinates of point reflectors
        all_point_y: list of corresponding y-coordinates of point reflectors
    output:
        radar_response: 2d complex numpy array, the echoed signals
    """
    assert(len(all_point_x) == len(all_point_y))
    num_points = len(all_point_x)

    radar_response = np.zeros((num_range_bins, num_channels), dtype = np.complex128)    
    for i in range(num_points):
 
        point_x = all_point_x[i]
        point_y = all_point_y[i]
        radar_response += add_point(point_x, point_y)

    return radar_response

def get_radar_image_pairs(radar_response):
    """
    input: 
        radar_response: 2d complex numpy array, the echoed signals
    output:
        polar_full: 3d-numpy array, log magnitude of full array results
        polar_partial_mag_phase 3d-numpy array, normalized log-magnitude times cos/sin of phase of sub-array results
    """
    # radar_response_original = copy.deepcopy(radar_response)
    polar_full = process_array(radar_response)
    polar_full = np.log10(np.abs(polar_full) + 10 **(-20))
    
    middle_response = radar_response[:, 500: 524]
    polar_middle = process_array(middle_response)
    # print('polar middle size', middle_response.shape)
    polar_middle = np.log10(np.abs(polar_middle) + 10 **(-20))

    polar_partial0_np = process_array(radar_response[:, 0: num_samples])

    polar_partial1_np = process_array(radar_response[:, num_channels - num_samples : num_channels])

    phase_cos0 = polar_partial0_np.real/(np.abs(polar_partial0_np) + 10 **(-20))
    phase_sin0 = polar_partial0_np.imag/(np.abs(polar_partial0_np) + 10 **(-20))
    phase_cos1 = polar_partial1_np.real/(np.abs(polar_partial1_np) + 10 **(-20))
    phase_sin1 = polar_partial1_np.imag/(np.abs(polar_partial1_np) + 10 **(-20))
    
    # phase1 = polar_partial1_np.imag/polar_partial1_np.real
    log_mag0 = np.log10(np.abs(polar_partial0_np) + 10 ** (-20))
    log_mag0 = (log_mag0 - np.min(np.min(log_mag0))) / (np.max(np.max(log_mag0)) - np.min(np.min(log_mag0)))
    log_mag1 = np.log10(np.abs(polar_partial1_np) + 10 ** (-20))
    log_mag1 = (log_mag1 - np.min(np.min(log_mag1))) / (np.max(np.max(log_mag1)) - np.min(np.min(log_mag1)))
    polar_partial_np = np.dstack([log_mag0 * phase_cos0, log_mag0* phase_sin0, log_mag1* phase_cos1, log_mag1* phase_sin1])
    polar_partial_np = polar_partial_np.transpose(2, 0, 1)

    output = {
            'polar_full': polar_full[np.newaxis, :, :],
            'polar_partial_mag_phase': polar_partial_np,
            'polar_middle': polar_middle[np.newaxis, :, :],
       }
    return output

def norm01_2d(data):
    """
    input: 2d-numpy array
    output: normalized input in range [0, 1] 
    """
    min_data = np.min(np.min(data))
    max_data = np.max(np.max(data))
    return (data - min_data)/ (max_data - min_data)

def get_vline(start_x, start_y, depth):
    """
    inputs:
        start_x: x-coordinate of the first point reflector in a vertical line
        start_y: y-coordinate of the first point reflector in a vertical line
        depth: integer, number of point reflectors in the vertical line
    outputs:
        all_point_x: x-coordinates of all the points in a vertical line
        all_points_y: corresponding y-coordinates of all the points in a vertical line
    """
    if start_x > max_rng * 0.8:
        start_x = max_rng * 0.8
    if start_x < - max_rng * 0.8:
        start_x = - max_rng * 0.8
    start_y = np.min([start_y, max_rng * 0.9 - depth * rng_res * 1.5])
    all_point_y = np.arange(depth) * rng_res * 1.5 + start_y
    all_point_x = np.ones(all_point_y.shape) * start_x
    return all_point_x, all_point_y

def get_random_point(num_points = 1): 
    """
    input:
        num_points: the number of random points
    outputs:
        all_point_x: x-coordinates of the random points
        all_points_y: corresponding y-coordinates of the random points
    """
    all_ranges = np.random.randn(num_points) * max_rng * 0.3
    all_ranges = all_ranges + max_rng / 2
    all_ranges[all_ranges< max_rng * 0.2] = max_rng * 2
    all_ranges[all_ranges > max_rng * 0.8] = max_rng * 0.8
    all_angles = np.random.randn(num_points)
    all_angles[all_angles > 0.8 * np.pi] = 0.8 * np.pi
    all_angles[all_angles < - 0.8 * np.pi] = - 0.8 * np.pi
    all_point_x = all_ranges * np.cos(all_angles)
    all_point_y = all_ranges * np.sin(all_angles)
    return all_point_x, all_point_y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description='set mode for data generation')
    parser.add_argument('--mode', type = str, default = 'debug')
    parser.add_argument('--directory', type = str, default = 'mooooo')
    parser.add_argument('--sample_type',type = str, default = 'vline' )
    parser.add_argument('--max_num_points', type = int, default = 1)
    parser.add_argument('--num_scenes', type = int, default = 1)
    parser.add_argument('--mix', type = int, default = 0)
    args = parser.parse_args()

    mode = args.mode
    max_num_points = args.max_num_points
    num_scenes = args.num_scenes
    print(mode)

    data_dir = os.path.join('cloud_data', args.directory, mode)
    os.makedirs(data_dir, exist_ok = True)
    

    for n in range(num_scenes):
        all_point_x = []
        all_point_y = []
        
        num_objects = args.max_num_points
        if args.mix == 1:
            start_x = np.random.rand() * 2 * max_rng - max_rng
            start_y = np.random.rand() * max_rng
            point_x, point_y = get_vline(start_x, start_y, 75)
            all_point_x = np.hstack([all_point_x, point_x])
            all_point_y = np.hstack([all_point_y, point_y])

            line_min_rng = np.sqrt(start_x ** 2 + start_y **2)

            point_rng = line_min_rng + np.random.rand() * 50 * rng_res
            point_ang = np.random.rand() * np.pi - np.pi / 2
            point_x = (np.cos(point_ang) * point_rng)
            point_y = np.sin(point_ang) * point_rng

            all_point_x = np.hstack([all_point_x, point_x])
            all_point_y = np.hstack([all_point_y, point_y])
            num_objects = num_objects - 2
        # if num_objects < 0:
        #     continue 

        for i in range(num_objects):
            if args.sample_type == 'mix':
                a = np.random.rand()
                if a < 0.5:
                    object_to_add = 'points'
                else:
                    object_to_add = 'vline'
            else:
                object_to_add = args.sample_type

            if object_to_add == 'points':
                point_x, point_y = get_random_point(1)
            elif object_to_add == 'vline':
                start_x = np.random.rand() * 2 * max_rng - max_rng
                start_y = np.random.rand() * max_rng
                point_x, point_y = get_vline(start_x, start_y, 75)

            all_point_x = np.hstack([all_point_x, point_x])
            all_point_y = np.hstack([all_point_y, point_y])

        scene_name = args.sample_type + '_' + str(num_objects) +'_objects_'+ str(n) + '.npz'
        save_path = os.path.join(data_dir, scene_name)

        raw_data = get_scene_raw_data(all_point_x, all_point_y)
        processed = get_radar_image_pairs(raw_data)
        np.savez_compressed(save_path, all_point_x = all_point_x, 
                all_point_y = all_point_y,
                polar_full = processed['polar_full'],
                polar_partial_mag_phase = processed['polar_partial_mag_phase'],
                polar_middle = processed['polar_middle']


                )

        if n % 10 == 0:
            print(n)