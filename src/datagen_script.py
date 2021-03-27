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

if __name__ == '__main__':
    object_types = ['points', 'vline']

    