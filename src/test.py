import numpy as np 
import torch
import json
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import os
import argparse
from torch.optim import lr_scheduler
import torch.utils.data as data

import newmodels
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

if __name__ == '__main__':
    print('finished importing stuff')
    parser = argparse.ArgumentParser(description='parse arguments for testing')
    parser.add_argument('--data_directory', nargs='+', default=['cloud_data', 'mooooo', 'debug'])
    parser.add_argument('--model_path',  type = str, default = 'trained_model.pt')
    parser.add_argument('--show_every', type = int, default = 1)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = args.model_path

    model = torch.load(model_path, map_location=device)
    model.to(device)
    # model.train()
    print('loaded model: ' + model_path)
    summary(model, (4, 512, 1024))
    
    data_dir = os.path.join(*args.data_directory)
    print('data directory: ', data_dir)
    net_input_name = 'polar_partial_mag_phase'
    target_name = 'polar_full'
    data_list = os.listdir(data_dir)[0:-1]

    ###############################################################################
    # test model
    ###############################################################################
    show_figs = True
    check_all = False
    nums_examined = 0
    show_every = args.show_every

    mse = nn.MSELoss(reduction = 'sum')
    mydataset = mydata.PointDataSet(data_dir, data_list, pre_processed = True)
    mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= False, num_workers= 4)
    total_error = 0
    for batch_idx, sample in enumerate(mydataloader):    
        target = sample[target_name]
        target = mydata.norm01(target)
        net_input = sample[net_input_name]
        net_output = model(net_input)

        loss = mse(net_output, target)

        total_error += loss.item()

        if nums_examined % show_every == 0:
            print('showing sample ', sample['file_path'])
            print('squared error:', loss)

            mydata.plot_labeled_grid(net_input, net_output, target)

            plt.show()
        nums_examined += 1


    print('average squared error: ', total_error / nums_examined)