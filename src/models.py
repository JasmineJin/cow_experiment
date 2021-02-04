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
import datagen

class SuperDense(nn.Module):
    def __init__(self, depth = 6, use_bias = True):
        super(SuperDense, self).__init__()
        lin1 = nn.Sequential(nn.Linear(4 * datagen.num_samples, datagen.num_channels, bias = use_bias), 
                                    nn.LeakyReLU(),
                                    )
        lin2 = nn.Sequential(nn.Linear(datagen.num_channels, datagen.num_channels, bias = use_bias),
                                    nn.Tanh(),
                                    )
        my_layers = [lin1, lin2]
        for i in range(depth):
            my_layers += [nn.Sequential(nn.Linear(datagen.num_channels, datagen.num_channels, bias = use_bias),
                            nn.LeakyReLU(),
                            )]
        my_layers += nn.Sequential(nn.Linear(datagen.num_channels, datagen.num_channels, bias = use_bias),
                                    nn.Sigmoid(),
                                    )
        self.model = nn.Sequential(*my_layers)
    def forward(self, x):
        return self.model(x).unsqueeze(1)

class UnetLin(nn.Module):
    """dense layer and then unet"""

    def __init__(self, input_nc = 1, output_nc= 1, num_downs= 8, ngf=64, norm_layer=nn.BatchNorm2d, final_act = None, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        
        super(UnetLin, self).__init__()
        # construct the linear structure
        self.lin1 = nn.Sequential(nn.Linear(4 * datagen.num_samples, datagen.num_channels, bias = use_bias), 
                                    nn.LeakyReLU(),
                                    )
        self.lin2 = nn.Sequential(nn.Linear(datagen.num_channels, datagen.num_channels, bias = use_bias),
                                    nn.LeakyReLU(),
                                    )
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias, final_act = final_act)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias, final_act = final_act)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=1, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        x1 = self.lin1(input)
        # return self.lin2(x1).unsqueeze(1)
        x2 = self.lin2(x1)
        x3 = x2.unsqueeze(1)
        return self.model(x3)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, final_act = None, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias, final_act = final_act)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias, final_act = final_act)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False, final_act = None):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if final_act == None:
                up = [uprelu, upconv]
            else:
                final_act_fun = final_act()
                up = [uprelu, upconv, final_act_fun]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGeneratorCustom(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, final_act = None, use_bias = True, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorCustom, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlockCustom(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_bias = use_bias, final_act = final_act)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlockCustom(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias, final_act = final_act)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlockCustom(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlockCustom(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        unet_block = UnetSkipConnectionBlockCustom(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)
        self.model = UnetSkipConnectionBlockCustom(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_bias = use_bias, final_act = final_act)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlockCustom(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_bias = True, use_dropout=False, final_act = None):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlockCustom, self).__init__()
        self.outermost = outermost
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Sequential(nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=2, dilation = 2, bias=use_bias, padding_mode = 'circular'),
                             nn.MaxPool2d(2))
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if final_act == None:
                up = [uprelu, upconv]
            else:
                final_act_fun = final_act()
                up = [uprelu, upconv, final_act_fun]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num_downs = 5, use_bias = True):
        
        super(MyModel, self).__init__()
        sub_model = MyModelSkipModule(mid_channels, mid_channels, use_bias = use_bias)
        for i in range(num_downs):
            sub_model = MyModelSkipModule(mid_channels, mid_channels, submodule = sub_model, use_bias = use_bias)
        self.model = MyModelSkipModule(in_channels, out_channels, mid_channels, submodule = sub_model, use_bias = use_bias)
        # model  = [down, up]
        # self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
        # return self.model(x)

class MyModelSkipModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, down_kernel_size = 4, down_dilation = 2, down_padding = 3,
                    up_kernel_size = 3, up_dilation = 1, up_padding = 1, submodule = None, use_bias = True):
        
        super(MyModelSkipModule, self).__init__()
        if mid_channels == None:
            mid_channels = in_channels
            
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = down_kernel_size, dilation = down_dilation, padding = down_padding, padding_mode = 'circular', bias = use_bias)
        self.relu1 = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.down = nn.MaxPool2d(2, return_indices = True)
        self.up = nn.MaxUnpool2d(2)
        # if submodule == None:
        #     mid_channels = mid_channels
        # else:
        #     mid_channels = mid_channels * 2
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size = up_kernel_size, padding = up_padding, dilation = up_dilation, padding_mode = 'circular', bias = use_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sig = nn.Sigmoid()
        self.submodule = submodule
        # model  = [down, up]
        # self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_down, down_indices = self.down(x)
        if self.submodule != None:
            x_up = self.up(self.submodule(x_down), down_indices)
        else:
            x_up = self.up(x_down, down_indices)
        x_up = self.conv2(x_up)
        x_up = self.bn2(x_up)
        return self.sig(x_up)
        # return self.model(x)



if __name__ == '__main__':
    device = torch.device('cpu')
    mynet = SuperDense()
    myinput = torch.rand(256, 64)
    myoutput = mynet(myinput)
    print(myoutput.size())

    summary(mynet, (256, 64))
