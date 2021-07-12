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

def get_music_spectrum():
    return 0
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    
    M = 16 # number of sensors
    source_power = 1
    noise_power = 0.1

    frequency_vector = np.random.rand(5) * np.pi - 0.5 * np.pi
    N = len(frequency_vector) # number of sources
    
    # steering_vector1 = get_steering_vector(frequency, M)
    # print(steering_vector1.shape)
    manifold_matrix = np.zeros([M, len(frequency_vector)], dtype = np.complex)

    for i in range(len(frequency_vector)):
        frequency = frequency_vector[i]
        steer = get_steering_vector(frequency, M)
        manifold_matrix[:, i] = steer[:, 0]
    
    print(manifold_matrix.shape)
    
    T = 200

    cov_estimate = np.zeros([M, M], dtype = np.complex)
    for i in range(T):
        source_value = (np.random.randn(N)  + 1j * np.random.randn(N)) * source_power / np.sqrt(2)
        received_values = np.dot(manifold_matrix, source_value[:, np.newaxis]) + (np.random.randn(M)  + 1j * np.random.randn(M)) * noise_power / np.sqrt(2)
        # print(received_values.shape)
        cov_estimate += np.dot(received_values, np.conjugate(received_values.T))
    
    print(cov_estimate.shape)
    cov_estimate = cov_estimate / T
    eig_values, eig_vectors = np.linalg.eig(cov_estimate)

    plt.figure()
    plt.plot(np.abs(eig_values))
    plt.title('eigen values')
    largest_eig = np.max(np.abs(eig_values))
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

    plt.figure()
    plt.plot(all_frequencies, np.abs(spectrum_music)/ np.max(np.abs(spectrum_music)))
    plt.vlines(frequency_vector, ymin = 0, ymax  =1.1, color = 'r')
    plt.title('music spectrum')
    plt.show()

    # x1 = np.array([[1 + 1j, 1 + 2j], [2 + 1j, 2 + 2j]])

    # print(x1)
    # print(np.conjugate(x1.T))

    
