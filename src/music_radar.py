import numpy as np
import scipy as sp
from numpy.fft import fft, ifft, fftshift, ifftshift
import os
import scipy as sp
import copy

import data_generation as gen

def get_steering_vector(aoa, N):
    frequency = np.pi * np.cos(aoa)
    steer_arguments = np.arange(N) * frequency
    steering_vector = np.exp(1j * steer_arguments)
    return steering_vector[:, np.newaxis]

def get_music_spectrum(cov_estimate, M):
    eig_values, eig_vectors = np.linalg.eig(cov_estimate)

    largest_eig = np.max(np.abs(eig_values))
    plt.figure()
    plt.plot(np.abs(eig_values))
    plt.title('eigenvalues')

    noise_space = eig_vectors[:, np.abs(eig_values) <= 0.01 * largest_eig]
    print("noise space shape", noise_space.shape)
    noise_multiply = np.dot(noise_space, np.conjugate(noise_space.T))
    all_frequencies = np.arange(1000) / 1000 * np.pi - np.pi / 2

    spectrum_music = np.zeros(len(all_frequencies))
    for i in range(len(all_frequencies)):
        frequency = all_frequencies[i]
        steer = get_steering_vector(frequency, M)
        
        power = np.dot(np.dot(np.conjugate(steer.T), noise_multiply), steer)
        # print(power.shape)
        # break
        spectrum_music[i] = 1 / np.abs(power[0, 0])
    return all_frequencies, spectrum_music

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 5
    aoa_vector = np.random.rand(N) * np.pi - np.pi / 2
    common_range = 200 * gen.rng_res
    
    point_x = common_range * np.cos(aoa_vector) 
    point_y = common_range * np.sin(aoa_vector)#, point_y = gen.get_random_point(N)
    
    array_response = gen.get_scene_raw_data(point_x, point_y)

    range_window = np.hanning(gen.num_range_bins)
    range_window_mtx = np.diag(range_window)

    num_channels_to_process = array_response.shape[1]
    channel_window = np.hanning(num_channels_to_process)
    channel_window_mtx = np.diag(channel_window)

    range_processed = fft(np.dot(range_window_mtx, array_response), axis = 0, norm= 'ortho')
    
    one_range = range_processed[200, :]
    angle_profile = fft(one_range, norm = 'ortho')

    plt.figure()
    plt.title('range profile')
    plt.plot(np.abs(range_processed[:, 0]))

    angle_vector = np.arange(len(angle_profile)) / len(angle_profile) * 2 - 1
    angle_profile = np.roll(angle_profile, len(angle_profile) // 2)
    plt.figure()
    plt.title('angle profile')
    plt.plot(angle_vector, np.abs(angle_profile))
    plt.vlines(np.cos(aoa_vector), ymin = 0, ymax = np.max(np.abs(angle_profile)), color = 'r')
    # plt.show()

    range_processed_noisy = range_processed + (np.random.randn(*range_processed.shape) + 1j * np.random.randn(*range_processed.shape)) 
    cov_estimate = np.dot(np.conjugate(range_processed.T), range_processed) / 1024
    print('estimated cov shape', cov_estimate.shape)
    # corr_mat = np.dot(np.transpose(range_processed), range_processed)
    all_freq, music_spectrum = get_music_spectrum(cov_estimate, 1024)

    plt.figure()
    plt.title('music angle profile')
    plt.plot(all_freq, music_spectrum)
    plt.vlines(np.cos(aoa_vector), ymin = 0, ymax = 1.1 * np.max(np.abs(music_spectrum)), color = 'r')
    plt.show()

