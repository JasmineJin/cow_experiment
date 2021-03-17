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
import newmodels
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
# set up G model
# set up D model
# define loss functions
# set up training and validation dataloaders
# set up optimizers
# set up scheduler (nope)
# set up summary writer
# other training configs
# training loop
# -- alternate between training D and trainging G
# -- go through training set
# -- go through validation set
# -- log data into writer
# -- print stuff
# -- check model updates
# save models
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
    elif args.final_act == 'relu':
        final_act = nn.ReLU
    elif args.final_act == 'leaky':
        final_act = nn.LeakyReLU
    
    else:
        final_act = None

    ##########################################################################
    # set up model
    ##########################################################################
    # if args.net_type == 'unet2d':
    #     model = models.UnetGenerator(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias, final_act = final_act)
    #     # model = models.UNet2D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device, sig_layer = True)
    # elif args.net_type == 'unet1d':
    #     model = models.UnetGenerator1D(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias)
    #     # model = models.UNet1D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device)
    # elif args.net_type == 'unetvh':
    #     model = models.UnetGeneratorVH(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias)
    # elif args.net_type == 'unet_custom':
    #     model = models.UnetGeneratorCustom(input_nc = args.in_channels, output_nc = args.out_channels, num_downs = args.depth, ngf = args.mid_channels, use_bias = args.use_bias, final_act = final_act)
    # elif args.net_type == 'tinynet':
    #     model = models.MyModel(in_channels = args.in_channels, out_channels = args.out_channels, mid_channels = args.mid_channels, num_downs = args.depth, use_bias = args.use_bias)
    # elif args.net_type == 'new_model':
    A = newmodels.MultiFilter(args.in_channels, args.out_channels, args.mid_channels, args.depth, args.use_bias, final_act)
    # A = newmodels.MultifilterSame(args.in_channels, args.mid_channels, args.out_channels, args.depth, args.use_bias, final_act, use_dropout = True)
    D = newmodels.Critic(args.out_channels * 2, args.mid_channels, 512, use_bias = args.use_bias)
    # elif args.net_type == 'cnn':
    #     model = newmodels.MultifilterSame(args.in_channels, args.mid_channels, args.out_channels, args.depth, args.use_bias, final_act)
    # else:
    #     print(args.net_type)
    #     raise NotImplementedError('model not implemented')
    
    A.to(device)
    D.to(device)
    print('created model: ')
    # print(model)

    #########################################################################
    # define loss function 
    #########################################################################
    lambda1 = args.lambda1
    mse = nn.MSELoss()
    l1_loss = nn.L1Loss()
    bce = nn.BCELoss()

    
    #########################################################################
    # set up training and validation dataloaders
    #########################################################################
    net_input_name = args.net_input_name
    target_name = args.target_name
    data_dir_train = os.path.join(*args.train_data_dir)
    print('taking training data from directory:', data_dir_train)
    data_list_train = os.listdir(data_dir_train)
    train_dataset = mydata.PointDataSet(data_dir_train, data_list_train[0: args.num_train], net_input_name, target_name, args.pre_processed)
    train_dataloader = data.DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle= args.shuffle_data, num_workers= 1)

    if len(args.val_data_dir) == 0:
        val_dataloader = train_dataloader
        print('using the same dataset for training and validation')
    else:
        data_dir_val = os.path.join(*args.val_data_dir)
        print('taking validation data from directory:', data_dir_val)
        data_list_val = os.listdir(data_dir_val)
        val_dataset = mydata.PointDataSet(data_dir_val, data_list_val[0: args.num_val], args.net_input_name, args.target_name, args.pre_processed)
        val_dataloader = data.DataLoader(val_dataset, batch_size = 1, shuffle= False, num_workers= 1)

    ##########################################################################
    # set up optimizer
    ##########################################################################
    if args.optimizer_type == 'Adam':
        optimizerA = optim.Adam(A.parameters(), lr = args.learning_rateA, betas = (args.momentum, 0.999))
        optimizerD = optim.Adam(D.parameters(), lr = args.learning_rateD, betas = (args.momentum, 0.999))
    elif args.optimizer_type == 'sgd':
        optimizerA = optim.SGD(A.parameters(), lr = args.learning_rateA, momentum = args.momentum)
        optimizerD = optim.SGD(D.parameters(), lr = args.learning_rateD, momentum = args.momentum)
    else:

        raise NotImplementedError('optimizer option not implemented')

    # ##########################################################################
    # # set up scheduler
    # ##########################################################################
    # scheduler = lr_scheduler.StepLR(optimizerA, step_size= args.scheduler_stepsize, gamma= args.scheduler_gamma)

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
    thing_to_train_list = ['D', 'D', 'D', 'D', 'D', 'A', 'E']
    #########################################################################
    # training loop
    #########################################################################
    # curr_states = copy.deepcopy(model.state_dict())
    lowest_D_val_loss = float('inf')
    lowest_A_val_loss = float('inf')
    for e in range(num_epochs):
        D_train_loss = 0
        D_val_loss = 0
        A_train_loss = 0
        A_val_loss = 0
        num_D_samples = 1
        num_A_samples = 1

        # D.train()
        # A.eval()
        for thing_to_train in thing_to_train_list:
            for batch_idx, sample in enumerate(train_dataloader):        
                target = sample[target_name]#[:, 0, :, :].unsqueeze(1)
                if args.norm:
                    target = mydata.norm01(target)
                    # target = mydata.quantizer(target, 0, 1, 2 ** args.quantize)
                target = target.to(device)
                # if args.train_auto:
                #     net_input = target
                # else:
                net_input = sample[net_input_name]
                if args.norm:
                    net_input = mydata.norm01(net_input)
                        # net_input = mydata.quantizer(net_input, 0, 1, 2 ** args.quantize)
                net_input = net_input.to(device)
                
                # check which model we are training
                if thing_to_train == 'D':
                    D.train()
                    A.eval()
                    #forward
                    fake_output = A(net_input)
                    fake_together = torch.cat([fake_output, target], 1)
                    score_fake = D(fake_together.detach())
                    real_together = torch.cat([target, target], 1)
                    score_real = D(real_together.detach())
                    D_loss = 0.5 * bce(score_fake, torch.zeros(score_fake.size())).to(device) + 0.5 + bce(score_real, torch.ones(score_real.size()).to(device))
                    # backward
                    optimizerA.zero_grad()
                    optimizerD.zero_grad()
                    D_loss.backward()
                    # update
                    optimizerD.step()
                    # update epoch loss
                    D_train_loss += (D_loss.item() - D_train_loss) / (num_D_samples)
                    num_D_samples += 1

                elif thing_to_train == 'A': 
                    A.train()
                    D.eval()
                    #forward
                    fake_output = A(net_input)
                    fake_together = torch.cat([fake_output, target], 1)
                    fake_score = D(fake_together)
                    A_loss = lambda1 * bce(fake_output, target) + bce(fake_score, torch.ones(fake_score.size()).to(device))
                    
                    optimizerA.zero_grad()
                    optimizerD.zero_grad()
                    A_loss.backward()
                    # update
                    optimizerA.step()
                    # print('A loss:', A_loss)
                    # update epoch loss
                    A_train_loss += (A_loss.item() - A_train_loss) / num_A_samples#+= (D_loss.item() - D_train_loss) / num_train_samples
                # elif thing_to_train == 'E':
                #     # validate

        # # update lowest_val_loss
        # if val_loss <= lowest_val_loss:
        #     lowest_val_loss = val_loss

        # log data into writer 
        if e % log_every == 0:
            # img_grid = mydata.get_output_target_image_grid(fake_output, target, target_name)
            # writer.add_image('output and target pair after epoch ' + str(e), img_grid)
            writer.add_scalar('A', A_train_loss, e)
            writer.add_scalar('D', D_train_loss, e)
            # if args.show_image:
            #     mydata.display_data(target, myoutput, net_input, target_name, net_input_name)

        # print stuff and display stuff 
        if e % print_every == 0:
            print('epoch-{}; D_loss: {}; A_loss: {}'
                .format(e, D_train_loss, A_train_loss))

            if args.show_image:
                # for batch_idx, sample in enumerate(val_dataloader):
                #     # net_input = 
                #     if args.train_auto:
                #         net_input = sample[target_name]
                #     else:
                #         net_input = sample[net_input_name]
                #     if args.norm:
                #         net_input = mydata.norm01(net_input)
                #     net_input = target.to(device)
                #     fake_output = A(net_input)
                img_grid = mydata.get_output_target_image_grid(fake_output, target, target_name)
                writer.add_image('output and target pair after epoch ' + str(e), img_grid)
                mydata.matplotlib_imshow(img_grid, 'output and target')
                print('showing ', sample['file_path'])
                plt.show()
                # mydata.display_data(target, myoutput, net_input, target_name, net_input_name)

        # # check model updates
        # if check_every != 0 and e % check_every == 0:
        #     everything_is_good = True
        #     new_gradients = [param.grad for param in model.parameters()]
        #     for grad in new_gradients:
        #         if grad == None:
        #             print(e)
        #             raise ValueError('found some grad none')
        #             # everything_is_good = False

        #     new_states = copy.deepcopy(model.state_dict())
        #     for state_name in new_states:
        #         old_state = curr_states[state_name]
        #         new_state = new_states[state_name]
        #         if torch.equal(old_state, new_state):
        #             print(state_name, 'not changed')
        #             everything_is_good = False
        #     print('everything is good?', everything_is_good)
        #     curr_states = new_states

    writer.close()

    ##############################################################################
    # save model
    ##############################################################################
    torch.save(A, args.model_name + 'A_' + args.arg_file + '.pt')
    torch.save(D, args.model_name + 'D_' + args.arg_file + '.pt')