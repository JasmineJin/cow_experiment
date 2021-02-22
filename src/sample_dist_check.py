import numpy as np
import matplotlib.pyplot as plt
import train
import numpy as np
import scipy as sp
from numpy.fft import fft, ifft, fftshift, ifftshift
import os
import scipy as sp
import image_transformation_utils as trans
import datagen

all_ranges = np.random.randn(1000)
all_ranges = all_ranges + datagen.max_rng / 2
all_ranges[all_ranges< datagen.max_rng * 0.2] = datagen.max_rng * 2
all_ranges[all_ranges > datagen.max_rng * 0.8] =datagen.max_rng * 0.8

plt.figure()
plt.hist(all_ranges, bins='auto')
plt.title('range distribution')
plt.show()

all_angles = np.random.randn(1000)
all_angles[all_angles > 0.8 * np.pi] = 0.8 * np.pi
all_angles[all_angles < - 0.8 * np.pi] = - 0.8 * np.pi

plt.figure()
plt.hist(all_angles, bins='auto')
plt.title('angle distribution')
plt.show()