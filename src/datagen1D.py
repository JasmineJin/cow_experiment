import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.fft import fft, ifft, fftshift, ifftshift
import os
import scipy as sp
import image_transformation_utils as trans

SIGNAL_DIM = 512
NSAMPLES = 12

def get_1d_raw_data(theta, magnitude):
    """
    return e ^ j theta n, theta in range (- pi to pi), has shape n_thetas x 1
    """
    n_vec = np.arange(SIGNAL_DIM)
    n_vec = n_vec[np.newaxis, :]
    x = np.exp(1j * (np.dot(theta, n_vec)))
    x = x * magnitude
    x = np.sum(x, 0)
    return x

def get_processed(x):
    x_0 = x[0: NSAMPLES]
    x_0_fft = fft(x_0, n = SIGNAL_DIM, norm = 'ortho')

    x_1 = x[len(x) - NSAMPLES : len(x)]
    x_1_fft = fft(x_1, n = SIGNAL_DIM, norm = 'ortho')

    window = np.hamming(SIGNAL_DIM)
    x_fft = fft(x * window, n = SIGNAL_DIM, norm = 'ortho')

    
    target = np.abs(x_fft)
    target = target[np.newaxis, :]
    # everything = np.dstack((np.abs(x_fft), np.abs(x_0_fft), np.abs(x_1_fft)))
    # all_max = np.max(np.max(np.max(everything)))
    # target = target / all_max

    net_input = np.vstack([x_0_fft.real, x_0_fft.imag, x_1_fft.real, x_1_fft.imag])
    # net_input = net_input / all_max

    return (target, net_input)


def get_processed_normalized(x):
    x_0 = x[0: NSAMPLES]
    x_0_fft = fft(x_0, n = SIGNAL_DIM, norm = 'ortho')

    x_1 = x[len(x) - NSAMPLES : len(x)]
    x_1_fft = fft(x_1, n = SIGNAL_DIM, norm = 'ortho')

    window = np.hamming(SIGNAL_DIM)
    x_fft = fft(x * window, n = SIGNAL_DIM, norm = 'ortho')

    
    target = np.abs(x_fft)
    target = target[np.newaxis, :]
    everything = np.dstack((np.abs(x_fft), np.abs(x_0_fft), np.abs(x_1_fft)))
    all_max = np.max(np.max(np.max(everything)))
    target = target / all_max

    net_input = np.vstack([x_0_fft.real, x_0_fft.imag, x_1_fft.real, x_1_fft.imag])
    net_input = net_input / all_max

    return (target, net_input)

def gen_data_random_theta(num_samples, min_num_theta,  max_num_theta, mode, data_dir = ''):
    data_dir = os.path.join(data_dir, mode)
    os.makedirs(data_dir, exist_ok = True)
    for n in range(num_samples):
        n_thetas = np.random.randint(min_num_theta, max_num_theta)
        magnitude = np.random.rand(n_thetas, 1)
        # x = np.zeros([1, SIGNAL_DIM], dtype = np.complex128)
        theta = (np.random.rand(n_thetas, 1) * np.pi * 2) -  np.pi
        # x = get_1d_raw_data(theta, magnitude)
        # target, net_input = get_processed(x)

        filepath = str(n) + '.npz'
        save_path = os.path.join(data_dir, filepath)
        np.savez_compressed(save_path, target = target, net_input = net_input, true_freqs = theta)
        # return x

if __name__ == '__main__':
    mode = 'train'
    max_num_theta = 10
    min_num_theta = 2
    num_samples = 10000 #10000

    theta_0 = np.random.rand() * np.pi * 2 - np.pi
    theta = np.arange(1000) * 0.1 * np.pi / SIGNAL_DIM + theta_0
    theta = theta[:, np.newaxis]
    magnitude = np.ones(theta.shape)
    print(magnitude.shape)

    # magnitude = np.random.rand(5, 1)
    x = get_1d_raw_data(theta, magnitude)
    print(x.shape)
    target, net_input = get_processed(x)

    plt.figure()
    plt.plot(target[0, :])
    plt.show()
    print(target.shape)
    # print(net_input.shape)

    data_dir = os.path.join('dataset', 'points1d')

    gen_data_random_theta(num_samples, min_num_theta, max_num_theta, mode, data_dir)
