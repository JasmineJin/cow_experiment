#############################################################################
# Lord have mercy. Christ graciously hear me. 
# Blessed Carlo Acutis, pray for me
# St. Jude, pray for me
# St. Joseph, pray for me
# Our Lady Seat of Wisdom, pray for me
# Holy Mary, Mother of God, pray for me
#############################################################################
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

import models
import data_manage as mydata

import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import datagen
from numpy.fft import fft, ifft

if __name__ == '__main__':
##############################################################################
# Order of things:
# check device
# parse arguments
# set up model
# define loss function
# set up training and validation dataloaders
# set up optimizer
# set up scheduler
# set up summary writer
# other training configs
# training loop
# -- go through training set
# -- go through validation set
# -- log data into writer
# -- print stuff
# -- check model updates
# save model
##############################################################################

    ##########################################################################
    # check device
    ##########################################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##########################################################################
    # parse arguments from file
    ##########################################################################
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--arg_file', help = 'arg file')
    args1 = parser1.parse_args()
    arg_file_path = os.path.join('train_args', args1.arg_file + '.json')

    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(open(arg_file_path)))
    args = parser1.parse_args(namespace=t_args)
    # print(vars(args))
    # print(args)
    ##########################################################################
    #  get the final layer's activation function
    ##########################################################################
    if args.final_act == 'sigmoid':
        final_act = nn.Sigmoid
    else:
        final_act = None

    ##########################################################################
    # set up model
    ##########################################################################
    if args.net_type == 'unet2d':
        model = models.UnetGenerator(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias, final_act = final_act)
        # model = models.UNet2D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device, sig_layer = True)
    elif args.net_type == 'unet1d':
        model = models.UnetGenerator1D(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias)
        # model = models.UNet1D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device)
    elif args.net_type == 'unetvh':
        model = models.UnetGeneratorVH(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias)
    # elif args.net_type == 'myunet':
    #     model = models.UNet2D(in_channels = args.in_channels, out_channels = args.out_channels, mid_channels = args.mid_channels)
    else:
        print(args.net_type)
        raise NotImplementedError
    
    model.to(device)
    print('created model: ')
    # print(model)

    #########################################################################
    # define loss function #TODO add option for l1 loss
    #########################################################################
    if args.loss_type == 'mse':
        myloss = nn.MSELoss(reduction = args.reduction)
    elif args.loss_type == 'log_mse':
        mse = nn.MSELoss(reduction = args.reduction)
        def myloss(output, target):
            return torch.log(mse(output, target))
    elif args.loss_type == 'log_ratio':
        def myloss(output, target):
            return torch.sum(torch.abs(torch.log(target / output)))
    elif args.loss_type == 'l1':
        def myloss(output, target):
            return torch.mean(torch.abs(output - target))
    elif args.loss_type == "bce":
        myloss = nn.BCELoss(reduction = args.reduction)
    else:
        raise NotImplementedError('loss function not implemented')
    
    #########################################################################
    # set up training and validation dataloaders
    #########################################################################
    net_input_name = args.net_input_name
    target_name = args.target_name
    data_dir_train = os.path.join(*args.train_data_dir)
    print('taking training data from directory:', data_dir_train)
    data_list_train = os.listdir(data_dir_train)
    train_dataset = mydata.PointDataSet(data_dir_train, data_list_train[0: args.num_train], net_input_name, target_name)
    train_dataloader = data.DataLoader(train_dataset, batch_size = 1, shuffle= args.shuffle_data, num_workers= 1)

    if len(args.val_data_dir) == 0:
        val_dataloader = train_dataloader
        print('using the same dataset for training and validation')
    else:
        data_dir_val = os.path.join(*args.val_data_dir)
        print('taking validation data from directory:', data_dir_val)
        data_list_val = os.listdir(data_dir_val)
        val_dataset = mydata.PointDataSet(data_dir_val, data_list_val[0: args.num_val], args.net_input_name, args.target_name)
        val_dataloader = data.DataLoader(val_dataset, batch_size = 1, shuffle= False, num_workers= 1)

    ##########################################################################
    # set up optimizer
    ##########################################################################
    if args.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    elif args.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
    else:
        raise NotImplementedError('optimizer option not implemented')

    ##########################################################################
    # set up scheduler
    ##########################################################################
    scheduler = lr_scheduler.StepLR(optimizer, step_size= args.scheduler_stepsize, gamma= args.scheduler_gamma)

    ##########################################################################
    # set up summmary writer
    ##########################################################################
    #
    writer_directory = os.path.join('runs', 'experiment_with_' + args.arg_file + args.exp_name)
    writer = SummaryWriter(writer_directory)

    ##########################################################################
    # other training configs
    ##########################################################################
    num_epochs = args.num_epochs
    print_every = args.print_every
    check_every = args.check_every
    log_every = args.log_every

    #########################################################################
    # training loop
    #########################################################################
    curr_states = copy.deepcopy(model.state_dict())
    lowest_val_loss = float('inf')
    for e in range(num_epochs):
        
        train_loss = 0
        val_loss = 0
        num_train_samples = 1
        num_val_samples = 1
        # go through training data
        model.train()
        for batch_idx, sample in enumerate(train_dataloader):
            target = sample[target_name]
            net_input = sample[net_input_name]
            target = target.to(device)
            net_input = net_input.to(device)
            #forward
            myoutput = model(net_input)
            loss = myloss(myoutput, target)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()
            # update epoch loss
            train_loss += (loss.item() - train_loss) / num_train_samples
            num_train_samples +=1
            if num_train_samples % args.refresh_every == 0:
                print(num_train_samples, train_loss, flush = True)

        # go through validation data
        # model.eval()
        for batch_idx, sample in enumerate(val_dataloader):
            target = sample[target_name]
            net_input = sample[net_input_name]
            target = target.to(device)
            net_input = net_input.to(device)
            #forward
            myoutput = model(net_input)
            lossval = myloss(myoutput, target)
            # update epoch loss
            # print('val loss: ', lossval)
            val_loss += (lossval.item() - val_loss) / num_val_samples
            num_val_samples +=1

        scheduler.step()

        # update lowest_val_loss
        if val_loss <= lowest_val_loss:
            lowest_val_loss = val_loss

        # log data into writer 
        if e % log_every == 0:
            # img_grid = mydata.get_output_target_image_grid(myoutput, target, target_name)
            # writer.add_image('output and target pair after epoch ' + str(e), img_grid)
            writer.add_scalar('validation loss', val_loss, e)
            writer.add_scalar('training loss', train_loss, e)
            # if args.show_image:
            #     mydata.display_data(target, myoutput, net_input, target_name, net_input_name)

        # print stuff and display stuff 
        if e % print_every == 0:
            print(e, 'train loss', train_loss)
            print(e, 'val loss', val_loss)
            if args.show_image:
                mydata.display_data(target, myoutput, net_input, target_name, net_input_name)

        # check model updates
        if check_every != 0 and e % check_every == 0:
            everything_is_good = True
            new_gradients = [param.grad for param in model.parameters()]
            for grad in new_gradients:
                if grad == None:
                    print(e)
                    raise ValueError('found some grad none')
                    # everything_is_good = False

            new_states = copy.deepcopy(model.state_dict())
            for state_name in new_states:
                old_state = curr_states[state_name]
                new_state = new_states[state_name]
                if torch.equal(old_state, new_state):
                    print(state_name, 'not changed')
                    everything_is_good = False
            print('everything is good?', everything_is_good)
            curr_states = new_states

    writer.close()

    ##############################################################################
    # save model
    ##############################################################################
    torch.save(model, args.model_name + '_' + args.arg_file + '.pt')