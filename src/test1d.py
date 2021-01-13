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
# import datagen
print('finished importing stuff')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = os.path.join('dataset', 'points1d', 'test')
model_path = os.path.join('models1d_trained', 'multi_point_model1d.pt')
model_info = torch.load(model_path)
model_weights = model_info['model_state_dict']
print('loaded weights')

model = models.UNet1D(in_channels= 4, out_channels=1, mid_channels= 16, depth = 6, kernel_size= 3, padding = 2, dilation= 2, device = device)
model.load_state_dict(model_weights)
model.to(device)
model.eval()
print('made model')

bce = nn.BCELoss(reduction = 'sum')
mse = nn.MSELoss(reduction= 'sum')
def custom_loss_fcn(output, target):
    """
    bce loss plus l1-norm
    """
    loss = bce(output, target) # + 0.0001 * torch.sum(torch.abs(output))
    return loss
print('defined loss function')

mydataset = mydata.Point1DDataSet(data_dir, os.listdir(data_dir))
mydataloader = data.DataLoader(mydataset, batch_size = 1, shuffle= True, num_workers= 1)
print('created dataloader')

net_input_name = 'net_input'
target_name = 'target'
show_data = True
total_loss = 0.
total_num = 0

print('beginning to enumerate')


num_samples = len(os.listdir(data_dir))
idx = np.random.randint(num_samples)

sample = mydataset.__getitem__(idx)

# for batchidx, sample in enumerate(mydataloader):
train_torch = sample[net_input_name]
target_torch = sample[target_name]
target_torch = target_torch.unsqueeze(0).to(device)
train_torch = train_torch.unsqueeze(0).to(device)

output = model(train_torch)
loss = custom_loss_fcn(output, target_torch)
total_loss += loss.item()
total_num += 1

if show_data:
    show_data = False
    print('showing sample number ', idx)
    output_np = output.cpu().detach().numpy()
    output_np = output_np[0, 0, :]
    target_np = target_torch.cpu().detach().numpy()
    target_np = target_np[0, 0, :]

    train_np = train_torch.cpu().detach().numpy()
    train_sample = train_np[0, 0, :] + 1j * train_np[0, 1, :]
    train_sample = np.abs(train_sample)

    plt.figure()
    plt.plot(train_sample)
    plt.title('net input')

    plt.figure()
    plt.plot(output_np)
    plt.title('output')

    plt.figure()
    plt.plot(target_np)
    plt.title('target')

    plt.show()

#     break

average_loss = total_loss / total_num
print('average loss: ',  average_loss)