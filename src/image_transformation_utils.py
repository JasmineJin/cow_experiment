import os
import sys
import scipy.io as spio
import scipy as sp
import numpy as np
from numpy.fft import fft,fftshift,ifft,ifftshift,fft2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
# from lib.utils import data_processing 
from PIL import Image

class Pixel(object):
    def __init__(self, x_lim= 0, y_lim = 0):
        self.x = x_lim
        self.y = y_lim
        self.value = 0
        self.num_values = 0

    def add_pix_value(self, value):
        self.value += value
        self.num_values += 1
    
    def get_pix_value(self):
        value = self.value #/ self.num_values
        return value
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

class PixelAngle(object):
    def __init__(self, x_lim= 0, y_lim = 0, value = 0):
        self.x = x_lim
        self.y = y_lim
        self.value = 0
        self.num_values = 0

    def add_pix_value(self, value):
        self.value += value
        self.num_values += 1
    
    def get_pix_value(self):
        if self.num_values == 0:
            return 0
        value = (self.value) % (np.pi) - np.pi/2
        return value
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

class Point(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

class PixelGrid(object):
    def __init__(self, num_rows, num_cols, x_extent = [-45, 45], y_extent = [0, 45]):
        self.grid = np.ndarray([num_rows, num_cols], dtype= Pixel)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.x_vector = np.linspace(x_extent[0], x_extent[1], num= num_cols)
        self.y_vector = np.linspace(y_extent[0], y_extent[1], num= num_rows)
        for x in range(num_rows):
            for y in range(num_cols):
                self.grid[x, y] = Pixel()
    

    def find_pixel_location(self, x, y):
        if x >= self.x_extent[1] or x < self.x_extent[0]:
            # print("x is out of range")
            return -1
        if y >= self.y_extent[1] or y < self.y_extent[0]:
            # print(" is out of range")
            return -1
        x_res = (self.x_extent[1] - self.x_extent[0]) / self.num_cols
        y_res = (self.y_extent[1] - self.y_extent[0]) / self.num_rows
        x_pix = int(np.floor((x - self.x_extent[0])/x_res))
        y_pix = int(np.floor((y - self.y_extent[0])/y_res))
        # if x_pix >= self.num_cols:
        #     x_pix = self.num_cols - 1
        # if y_pix >= self.num_rows:
        #     y_pix = self.num_rows - 1
        return (y_pix, x_pix)
    
    def add_value(self, x_pix, y_pix, value):
        pix = self.grid[x_pix, y_pix]
        pix.add_pix_value(value)
    
    def get_grid(self):
        value_grid = np.zeros([self.num_rows, self.num_cols])
        for x in range(self.num_rows):
            for y in range(self.num_cols):
                pix = self.grid[x, y]
                value_grid[x, y] = pix.get_pix_value()
        return value_grid

    def get_x_vector(self):
        return self.x_vector
    
    def get_y_vector(self):
        return self.y_vector

class PixelAngleGrid(object):
    def __init__(self, num_rows, num_cols, x_extent = [-45, 45], y_extent = [0, 45]):
        self.grid = np.ndarray([num_rows, num_cols], dtype= Pixel)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.x_vector = np.linspace(x_extent[0], x_extent[1], num= num_cols)
        self.y_vector = np.linspace(y_extent[0], y_extent[1], num= num_rows)
        for x in range(num_rows):
            for y in range(num_cols):
                self.grid[x, y] = PixelAngle()
    

    def find_pixel_location(self, x, y):
        if x >= self.x_extent[1] or x < self.x_extent[0]:
            # print("x is out of range")
            return -1
        if y >= self.y_extent[1] or y < self.y_extent[0]:
            # print(" is out of range")
            return -1
        x_res = (self.x_extent[1] - self.x_extent[0]) / self.num_cols
        y_res = (self.y_extent[1] - self.y_extent[0]) / self.num_rows
        x_pix = int(np.floor((x - self.x_extent[0])/x_res))
        y_pix = int(np.floor((y - self.y_extent[0])/y_res))
        return (y_pix, x_pix)
    
    def add_value(self, x_pix, y_pix, value):
        pix = self.grid[x_pix, y_pix]
        pix.add_pix_value(value)
    
    def get_grid(self):
        value_grid = np.zeros([self.num_rows, self.num_cols])
        for x in range(self.num_rows):
            for y in range(self.num_cols):
                pix = self.grid[x, y]
                value_grid[x, y] = pix.get_pix_value()
        return value_grid

    def get_x_vector(self):
        return self.x_vector
    
    def get_y_vector(self):
        return self.y_vector

def polar_to_rect(data_mtx, wavelength, num_channels, rng, img_height, img_width):
    d_effective = wavelength / 2
    psi_lim = 2 * np.pi * d_effective / wavelength 
    fov = np.arcsin(wavelength/d_effective/2)
    max_dyst = rng[-1] + 0.1
    x_lim = np.sin(fov) * max_dyst

    psi_vector = np.linspace(-psi_lim, psi_lim, num=num_channels)
    angle_cosine = wavelength / (2 * np.pi * d_effective) * psi_vector
    radar_grid = PixelGrid(img_height, img_width, x_extent = [- x_lim, x_lim], y_extent= [0 , max_dyst])

    value_normalized = data_mtx

    for x in range(len(rng)):
        for y in range(num_channels):
            # value = value_ang[x, y]
            value_n = value_normalized[x, y]
            cos_theta = angle_cosine[y]
            dyst = rng[x]
            horizontal_dyst = cos_theta * dyst
            vertical_dyst = np.sqrt(1 - cos_theta**2) * dyst
            pix_coord = radar_grid.find_pixel_location(horizontal_dyst, vertical_dyst)
            if pix_coord != -1:
                radar_grid.add_value(pix_coord[0], pix_coord[1], value_n)

    value_grid = radar_grid.get_grid()
    x_vector = radar_grid.get_x_vector()
    y_vector = radar_grid.get_y_vector()
    return (value_grid, x_vector, y_vector)