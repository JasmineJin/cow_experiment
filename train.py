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
        print('training epoch ', e)
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

            if num_div % 1000 == 0:
                print(num_div)

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

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train_batch_size', type = int, default = 4)
    parser.add_argument('--train_net', type = bool, default = False)
    parser.add_argument('--data_dir', type = str, default = '')
    # parser.add_argument('--data_dir_val', type = str, default = '')
    parser.add_argument('--net_input_name', type = str, default = '')
    parser.add_argument('--target_name', type = str, default = '')
    parser.add_argument('--learning_rate', type = float, default = 0.1)
    parser.add_argument('--num_train_epochs', type = int, default = 5)
    parser.add_argument('--model_name', type = str, default = '')
    args = parser.parse_args(['@train_args.txt'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = models.UNet2D(in_channels= 2, out_channels=1, mid_channels= 4, depth = 6, kernel_size= 3, padding = 2, dilation= 2, device = device, sig_layer = True)
    train_net = args.train_net
    print('training set to ', train_net)
    bce = nn.BCELoss(reduction = 'sum')
    mse = nn.MSELoss(reduction= 'sum')
    def custom_loss_fcn(output, target):
        # target = target + 10 ** (-30)
        # output = output + 10 ** (-30)
        min_target = torch.min(target)
        min_output = torch.min(output)
        loss = 1 * torch.sum(torch.abs(target - output)) + 0.1 * torch.abs(torch.sum(torch.abs(target - min_target)) - torch.sum(torch.abs(output - min_output)))
        # loss = mse(output, target) + 0.0001 * torch.sum(torch.abs(output))
        return loss

    train_bsz = args.train_batch_size
    data_dir_train = os.path.join('cloud_data', args.data_dir, 'train')
    data_list_train = os.listdir(data_dir_train)
    train_dataset = mydata.PointDataSet(data_dir_train, data_list_train)
    train_dataloader = data.DataLoader(train_dataset, batch_size = train_bsz, shuffle= True, num_workers= 4)

    data_dir_val = os.path.join('cloud_data', args.data_dir, 'val')
    data_list_val = os.listdir(data_dir_val)
    val_dataset = mydata.PointDataSet(data_dir_val, data_list_val)
    val_dataloader = data.DataLoader(val_dataset, batch_size = 1, shuffle= True, num_workers= 1)

    net_input_name = args.net_input_name
    target_name = args.target_name

    optimizer = optim.Adam(model.parameters(), lr = 0.9, weight_decay= 0)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size= 50, gamma= 0.5)

    num_epochs = 1000

    sample_idx = 0 #np.random.randint(len(os.listdir(data_dir_train)))
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
    print('l1 norm of output: ', torch.sum(final_output))
    output_np = final_output.cpu().detach().numpy()
    output_np = output_np[0, 0, :, :]
    target_np = target_torch.cpu().detach().numpy()
    target_np = target_np[0, 0, :, :]

    # plt.figure()
    # plt.imshow(target_np)
    # plt.title('target')

    # plt.figure()
    # plt.imshow(output_np)
    # plt.title('net output')

    # plt.show()

    experiment_name = 'single_point_experiment2d'    
    writer = SummaryWriter(experiment_name)
    checkpoint_save_path = os.path.join('single_point_model2d.pt')
    if train_net:
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay= 0)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size= 5, gamma= 0.5)
        num_epochs = args.num_train_epochs

        model = train_model(model, device, train_dataloader, val_dataloader, net_input_name, target_name, custom_loss_fcn, scheduler, optimizer, num_epochs, writer)

    torch.save({'epoch': num_epochs, 
            'batchsize': train_bsz,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'net_input_name': net_input_name,
            'target_name': target_name
            }, checkpoint_save_path)


    print('model saved')