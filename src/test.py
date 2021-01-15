import numpy as np
import matplotlib.pyplot as plt
import copy
import os

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import models
import data_manage as mydata
import train

print('finished importing stuff')

args = train.parse_train_args('@../train_polar_args.txt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = os.path.join('cloud_data', 'points', 'train')
model_path = 'single_point_model_psnr_polar_overfit.pt'
# model_path = os.path.join('models_trained', 'point_model2d_final.pt')
model_info = torch.load(model_path, map_location=device)
model_weights = model_info['model_state_dict']
print('loaded weights')

if args.net_type == 'unet2d':
    model = models.UNet2D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device, sig_layer = True)
elif args.net_type == 'unet1d':
    model = models.UNet1D(in_channels= args.in_channels, out_channels=args.out_channels, mid_channels= args.mid_channels, depth = args.depth, kernel_size= args.kernel_size, padding = args.padding, dilation= args.dilation, device = device)
else:
    print(args.net_type)
    raise NotImplementedError

# model = models.UNet2D(in_channels= 2, out_channels=1, mid_channels= 4, depth = 6, kernel_size= 3, padding = 2, dilation= 2, device = device)
model.load_state_dict(model_weights)
model.to(device)
model.eval()
print('made model')


# print('defined loss function')

mydataset = mydata.PointDataSet(data_dir, os.listdir(data_dir), args.net_input_name, args.target_name)
mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= True, num_workers= 1)
print('created dataloader')

net_input_name = args.net_input_name
target_name = args.target_name
show_data = True
total_loss = 0.
total_num = 0

print('beginning to enumerate')


num_samples = len(os.listdir(data_dir))
idx = 0 #np.random.randint(num_samples)

sample = mydataset.__getitem__(idx)

# for batchidx, sample in enumerate(mydataloader):
train_torch = sample[net_input_name]
target_torch = sample[target_name]
target_torch = target_torch.unsqueeze(0).to(device)
train_torch = train_torch.unsqueeze(0).to(device)

output = model(train_torch)
loss = train.custom_loss_fcn1(output, target_torch)
total_loss += loss.item()
total_num += 1

if show_data:
    show_data = False
    mydata.display_data(target_torch, output, train_torch, args.target_name, args.net_input_name)
    # print('showing sample number ', idx)
    # output_np = output.cpu().detach().numpy()
    # output_np = output_np[0, 0,  :, :]
    # target_np = target_torch.cpu().detach().numpy()
    # target_np = target_np[0, 0, :, :]

    # print('max value in output: ', np.max(np.max(output_np)))
    # print('min value in output: ', np.min(np.min(output_np)))
    # plt.figure()
    # plt.imshow(output_np)
    # plt.title('output')

    # plt.figure()
    # plt.imshow(target_np)
    # plt.title('target')

    # plt.show()


#     break

average_loss = total_loss / total_num
print('average loss: ',  average_loss)