import numpy as np 
import torch

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

import models
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import datagen
from numpy.fft import fft, ifft

def train_model(model, device, train_dataloader, val_dataloader, net_input_name, target_name, loss_fcn, scheduler, optimizer, num_epochs, writer):
    
    model = model.to(device)

    lowest_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    print('starting to train')
    for e in range(num_epochs):
        epoch_train_loss = 0.
        num_div = 0
        model.train()
        for batch_idx, sample in enumerate(train_dataloader):
            train_torch = sample[net_input_name]
            target_torch = sample[target_name]
            target_torch = target_torch.to(device)
            train_torch = train_torch.to(device)

            # print(net_input_name, train_torch.size())
            # print(target_name, target_torch.size())
            # print('target max: ', torch.max(target_torch))
            output = model(train_torch)
            loss =  loss_fcn(output, target_torch)
            # print('loss: ', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss +=  loss.item()
            num_div += 1

        if num_div > 0:
            epoch_train_loss = epoch_train_loss / num_div
        else:
            epoch_train_loss = float('inf')
        scheduler.step()
        writer.add_scalar('train loss', epoch_train_loss, e)
        
        epoch_val_loss = 0.
        num_div = 0
        
        print('train loss for epoch ', e, epoch_train_loss)
        model.eval()
        for batch_idx, sample in enumerate(val_dataloader):
            train_torch = sample[net_input_name]
            target_torch = sample[target_name]
            target_torch = target_torch.to(device)
            train_torch = train_torch.to(device)
            
            output = model(train_torch)
            loss =  loss_fcn(output, target_torch)

            epoch_val_loss +=  loss.item()
            num_div += 1

        if num_div > 0:
            epoch_val_loss = epoch_val_loss / num_div
        else:
            epoch_val_loss = float('inf')

        # if e % 10 == 0 or e < 10:
        print('val loss for epoch ', e, epoch_val_loss)
        if epoch_val_loss <= lowest_loss:
            lowest_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        writer.add_scalar('val loss', epoch_val_loss, e)

    model.load_state_dict(best_model_wts)
    return model

def overfit_model(model, device, net_input, target, loss_fcn, scheduler, optimizer, num_epochs):

    model = model.to(device)
    model.train()
    net_input = net_input.to(device)
    target = target.to(device)
    for e in range(num_epochs):
        output = model(net_input)
        loss = loss_fcn(output, target)
        if e % 100 == 0:
            print(e, loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = models.UNet1D(in_channels= 4, out_channels=1, mid_channels= 16, depth = 6, kernel_size= 3, padding = 2, dilation= 2, device = device)
    train_net = False

    bce = nn.BCELoss(reduction = 'sum')
    mse = nn.MSELoss(reduction= 'sum')
    def custom_loss_fcn(output, target):
        loss = mse(output, target) #+ 0.01 * torch.sum(torch.abs(output))
        loss = loss / torch.sum(target)
        return loss

    train_bsz = 4
    data_dir_train = os.path.join('dataset', 'points1d', 'train')
    data_list_train = os.listdir(data_dir_train)
    train_dataset = mydata.Point1DDataSet(data_dir_train, data_list_train)
    train_dataloader = data.DataLoader(train_dataset, batch_size = train_bsz, shuffle= True, num_workers= 4)

    data_dir_val = os.path.join('dataset', 'points1d', 'val')
    data_list_val = os.listdir(data_dir_val)
    val_dataset = mydata.Point1DDataSet(data_dir_val, data_list_val)
    val_dataloader = data.DataLoader(val_dataset, batch_size = 1, shuffle= True, num_workers= 1)

    net_input_name = 'net_input'
    target_name = 'target'

    optimizer = optim.Adam(model.parameters(), lr = 0.9, weight_decay= 0)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size= 50, gamma= 0.8)

    num_epochs = 1000

    sample_idx = np.random.randint(len(os.listdir(data_dir_train)))
    sample = train_dataset.__getitem__(sample_idx)
    train_torch = sample[net_input_name]
    target_torch = sample[target_name]
    target_torch = target_torch.unsqueeze(0)
    train_torch = train_torch.unsqueeze(0)

    model = overfit_model(model, device, train_torch, target_torch, custom_loss_fcn, scheduler, optimizer, num_epochs)
    print('finished overfitting')
    # model.cuda()
    # model_overfitted.eval()
    # train_torch.cuda()
    train_torch = train_torch.to(device)
    model.to(device)
    # model_overfitted.to(device)
    model.eval()
    final_output = model(train_torch)
    
    print('output size: ', final_output.size())
    
    output_np = final_output.cpu().detach().numpy()
    output_np = output_np[0, 0, :, ]
    target_np = target_torch.cpu().detach().numpy()
    target_np = target_np[0, 0, :, ]

    plt.figure()
    plt.plot(target_np)
    plt.title('target')

    plt.figure()
    plt.plot(output_np)
    plt.title('net output')

    plt.show()

    experiment_name = 'multi_point_experiment1D'    
    writer = SummaryWriter(experiment_name)
    checkpoint_save_path = os.path.join('models1d_trained', 'multi_point_model1d.pt')
    if train_net:
        optimizer = optim.Adam(model.parameters(), lr = 0.1, weight_decay= 0)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size= 5, gamma= 0.8)
        num_epochs = 40

        model = train_model(model, device, train_dataloader, val_dataloader, net_input_name, target_name, custom_loss_fcn, scheduler, optimizer, num_epochs, writer)

    torch.save({'epoch': num_epochs, 
            'batchsize': train_bsz,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'net_input_name': net_input_name,
            'target_name': target_name
            }, checkpoint_save_path)


    