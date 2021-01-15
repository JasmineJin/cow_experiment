import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
import torch.optim as optim
import os
import copy
from torchsummary import summary
import queue
import copy
from PIL import Image
from torchvision import transforms

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size = (3, 3), padding = (1, 1), dilation = (1, 1), stride = (1, 1), final_layer = False, bias = True):
        super(DoubleConv2D, self).__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
                # nn.BatchNorm2d(out_channels),
                # nn.Linear(),
            )
            conv_layers.append(double_conv1)
        else:
            double_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular', bias = bias),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            conv_layers.append(double_conv1)
        self.double_conv = nn.Sequential(*conv_layers)
    def forward(self, x):
        return self.double_conv(x)

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,  kernel_size = 3, padding = 2, dilation = 1, stride = 1, final_layer = False):
        super().__init__()
        conv_layers = []
        if not mid_channels:
            mid_channels = out_channels
        
        double_conv0 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, stride = stride, padding_mode= 'circular'),
            # nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        conv_layers.append(double_conv0)

        if final_layer:
            double_conv1 = nn.Sequential(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                # nn.BatchNorm1d(out_channels),
                # nn.Sigmoid(),
            )
            conv_layers.append(double_conv1)
        else:
            double_conv1 = nn.Sequential(
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, padding_mode= 'circular'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
            conv_layers.append(double_conv1)
        self.double_conv = nn.Sequential(*conv_layers)
    def forward(self, x):
        return self.double_conv(x)

class Up1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super(Up1D, self).__init__()
        self.conv = DoubleConv1D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)

    def forward(self, x1):
        x1 = F.interpolate(x1, scale_factor= 2)
        x = x1
        return self.conv(x)

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 1, dilation = 1, bias = True):
        super(Up2D, self).__init__()
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation, bias = bias)

    def forward(self, x1):
        x = F.interpolate(x1, scale_factor= 2)
        return self.conv(x)

class Down1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.conv = DoubleConv1D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation)
        
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class Down2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 2, out_channels =1, kernel_size = 3, padding = 1, dilation = 1, stride = 1, bias = True):
        super(Down2D, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv2D(in_channels, out_channels, in_channels, kernel_size= kernel_size, padding = padding, dilation =  dilation, stride =1, bias = bias)
        
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu')):
        super().__init__()
        self.down_layers = []
        self.up_layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.inlayer = DoubleConv1D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, stride = 2)
        # self.midlayer = DoubleConv1D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.outlayer = DoubleConv1D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, final_layer = True)

        # for d in range(depth):
        #     downlayer = Down1D(mid_channels * 2 ** d, mid_channels * (2** (d + 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
        #     self.down_layers.append(downlayer.to(device))
        
        # for d in range(depth):
        #     # print(d)
        #     uplayer = Up1D(mid_channels * (2 ** (depth - d)), mid_channels * (2**(depth - d - 1)), kernel_size= kernel_size, padding = padding, dilation = dilation)
        #     self.up_layers.append(uplayer.to(device))
        for d in range(depth):
            downlayer = Down1D(mid_channels, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.down_layers.append(downlayer.to(device))
        
        for d in range(depth):
            # print(d)
            uplayer = Up1D(mid_channels, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation)
            self.up_layers.append(uplayer.to(device))

    def forward(self, x):
        
        # x = torch.view(x.size(0), self.in_channels, -1)
        x = self.inlayer(x)
        # print(x.size())
        down_queue = queue.LifoQueue(maxsize= len(self.down_layers))
        for down in self.down_layers:
            down_queue.put(x)
            x = down(x)
            # print('down: ', x.size())
        # x = self.midlayer(x)
        for up in self.up_layers:
            x_queued = down_queue.get()
            x = up(x)
            x = x + x_queued
        x = self.outlayer(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels = 2, out_channels =1, mid_channels = 16, depth = 6, kernel_size = 3, padding = 2, dilation = 1, device = torch.device('cpu') , sig_layer = True, bias = True):
        
        super(UNet2D, self).__init__()
        self.inlayer = DoubleConv2D(in_channels, mid_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # self.midlayer = DoubleConv2D(mid_channels * 2** depth, mid_channels * 2 ** (depth - 1), kernel_size= kernel_size, padding = padding, dilation = dilation)
        self.down1 = Down2D(mid_channels, mid_channels * 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.down2 = Down2D(mid_channels * 2, mid_channels * 4, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.down3 = Down2D(mid_channels * 4, mid_channels * 8, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.down4 = Down2D(mid_channels * 8, mid_channels * 16, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.down5 = Down2D(mid_channels * 16, mid_channels * 32, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.down6 = Down2D(mid_channels * 32, mid_channels * 64, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)

        self.up6 = Up2D(mid_channels* 64, mid_channels * 32, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up5 = Up2D(mid_channels* 32, mid_channels * 16, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up4 = Up2D(mid_channels* 16, mid_channels * 8, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up3 = Up2D(mid_channels* 8, mid_channels * 4, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up2 = Up2D(mid_channels* 4, mid_channels * 2, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        self.up1 = Up2D(mid_channels* 2, mid_channels, kernel_size= kernel_size, padding = padding, dilation = dilation, bias = bias)
        # print(sig_layer)

        self.outlayer = DoubleConv2D(mid_channels, out_channels, mid_channels= mid_channels// 2, kernel_size= kernel_size, padding = padding, dilation = dilation, final_layer= sig_layer, bias = bias)
        self.drop = nn.Dropout(p = 0.5, inplace = False)
    def forward(self, x):
        # print(x.size())
        x = self.inlayer(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.up6(x6)
        # # print('up: ', x.size())
        x = self.up5(x )
        # # print('up: ', x.size())
        x = self.up4(x + x4)
        # # print('up: ', x.size())
        x = self.up3(x )
        # # print('up: ', x.size())
        x = self.up2(x + x2)
        x = self.drop(x)
        # print('up: ', x.size())
        x = self.up1(x)

        return self.outlayer(x)

class DenseDown1D(nn.Module):
    def __init__(self, signal_dim = 128, down_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim * in_filters
        self.down_sample_factor = down_sample_factor
        width = int(signal_dim * n_filters / self.down_sample_factor)

        self.in_linear = nn.Linear(height, width, bias=False)

    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, -1)
        return x

class DenseDown2D(nn.Module):
    def __init__(self, signal_dim = (64, 128), down_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim[0] * signal_dim[1] * in_filters
        self.down_sample_factor = down_sample_factor
        width = int(signal_dim[0] * signal_dim[1] * n_filters / (self.down_sample_factor **2))

        self.in_linear = nn.Linear(height, width, bias=False)
        self.out_dim = (signal_dim[0] //  down_sample_factor, signal_dim[1] // down_sample_factor)
    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, self.out_dim[0], self.out_dim[1])

        return x

class DenseUp1D(nn.Module):
    def __init__(self, signal_dim = 128, up_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim * in_filters
        self.up_sample_factor = up_sample_factor
        width = int(signal_dim * n_filters * self.up_sample_factor)

        self.in_linear = nn.Linear(height, width, bias=False)

    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, -1)
        return x

class DenseUp2D(nn.Module):
    def __init__(self, signal_dim = (64, 128), up_sample_factor = 4, in_filters = 4, n_filters=16, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.relu = nn.LeakyReLU()
        height = signal_dim[0] * signal_dim[1] * in_filters
        self.up_sample_factor = up_sample_factor
        width = int(signal_dim[0] * signal_dim[1] * n_filters * (self.up_sample_factor **2))

        self.in_linear = nn.Linear(height, width, bias=False)
        self.out_dim = (signal_dim[0] *  up_sample_factor, signal_dim[1] * up_sample_factor)
    def forward(self, inp):
        # x = inp
        x = inp.view(inp.size(0), -1)
        x = self.in_linear(x)
        x = self.relu(x)
        x = x.view(inp.size(0), self.n_filters, self.out_dim[0], self.out_dim[1])

        return x

class DenseConv1D(nn.Module):
    def __init__(self, signal_dim = 128, up_layers = 4, in_filters = 4, mid_channels = 2, out_channels =1, kernel_size = 3, padding = 2, dilation = 1):
        super().__init__()
        down_sample_factor = 2** up_layers
        n_filters = 2 ** up_layers * out_channels

        self.dense_layer = DenseDown1D(signal_dim, down_sample_factor, in_filters, mid_channels)
        self.conv_mid = DoubleConv1D(mid_channels, n_filters, n_filters // 2, kernel_size, padding, dilation)

        upsample_layers = []
        for u in range(up_layers):
            up = Up1D(out_channels * 2 ** (up_layers - u), out_channels * 2 **(up_layers -u - 1), kernel_size, padding, dilation)
            upsample_layers.append(up)
        
        self.upsample = nn.Sequential(*upsample_layers)
        self.conv_out = DoubleConv1D(out_channels, out_channels, n_filters // 2, kernel_size, padding, dilation, True)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.conv_mid(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    myconv = UNet2D(1, 1, mid_channels = 16, depth = 6, kernel_size= 3, padding = 2, dilation = 2, bias = True, sig_layer = False) #DoubleConv2D(1, 1)#

    mse = nn.MSELoss()
    def myloss(output, target):
        return torch.sum(torch.abs(output - target))
    
    myimg_path = '../cloud_data/cameraman.ppm'
    myimg = Image.open(myimg_path)
    myimg.show()
    print(myimg)
    # display()
    myimg_tensor = transforms.ToTensor()(myimg).unsqueeze(0).type(torch.float)
    print(myimg_tensor.size())
    print('max in myimg: ', torch.max(myimg_tensor))
    print('min in myimg: ', torch.min(myimg_tensor))

    myimg_pil = transforms.ToPILImage()(myimg_tensor.squeeze(0))
    myimg_pil.show()

    myconv = myconv.to(device)
    myinput = torch.rand(1, 1, 256, 512).to(device)
    mytarget = myinput#myimg_tensor.to(device) # torch.rand(1, 1, 256, 512).to(device)
    # summary(myconv, (1, 256, 512))
    
    myconv.train()
    num_iter = 0
    check_every = 10
    print_every = 10
    print(len(list(myconv.parameters())))
    # curr_params = copy.deepcopy(list(myconv.parameters())) #and b = list(model.parameters())
    curr_states = copy.deepcopy(myconv.state_dict())
    myoptimizer = optim.SGD(myconv.parameters(), lr = 0.01, momentum = 0.5)

    for i in range(num_iter):
        param = myconv.parameters()
        old_gradients = [param.grad for param in myconv.parameters()]
        # for j in range(len(list(myconv.parameters()))):
        #     old.gradient
        myoutput = myconv(myinput)
        loss = mse(myoutput, mytarget)
        myoptimizer.zero_grad()
        loss.backward()
        myoptimizer.step()
        
        if i % print_every == 0:
            print(i, 'loss', loss)

        
        if i % check_every == 0:
            everything_is_good = True
            new_gradients = [param.grad for param in myconv.parameters()]
            # print('old gradients:', old_gradients)
            # print('new gradients:', new_gradients)
            for grad in new_gradients:
                if grad == None:
                    raise ValueError('found grad none')
                    # everything_is_good = False

            new_states = copy.deepcopy(myconv.state_dict())
            for state_name in new_states:
                old_state = curr_states[state_name]
                new_state = new_states[state_name]
                if torch.equal(old_state, new_state):
                    # print(state_name, 'not changed')
                    everything_is_good = False
            print('everything is good?', everything_is_good)
            # curr_params = new_params

    
    
