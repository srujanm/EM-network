import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .block import *

class unetDown(nn.Module): # type 
    def __init__(self, opt, in_num, out_num, 
                 cfg_down={'pool_kernel': (1,2,2), 'pool_stride': (1,2,2), 'out_num': 1},
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':-1, 'has_dropout':0}, block_id=0):
        super(unetDown, self).__init__()
        self.opt = opt
        if opt[0]==0: # max-pool
            self.down = nn.MaxPool3d(cfg_down['pool_kernel'], cfg_down['pool_stride'])
        elif opt[0]==1: # resBasic
            self.down = blockResNet(unitResBasic, 1, in_num, cfg_down['out_num'], stride_size=cfg_down['pool_stride'], cfg=cfg_conv)

        if opt[1]==0: # vgg-3x3 
            self.conv = blockVgg(2, in_num, out_num, cfg=cfg_conv)
        elif opt[1]==1: # res-18
            self.conv = blockResNet(unitResBasic, 2, in_num, out_num, cfg=cfg_conv)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down(x1)
        return x1, x2

class unetUp(nn.Module):
    # in1: skip layer, in2: previous layer
    def __init__(self, opt, in_num, outUp_num, inLeft_num, outConv_num,
        cfg_up={'pool_kernel': (1,2,2), 'pool_stride': (1,2,2)},
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':0.005, 'has_dropout':0}, block_id=0):
        super(unetUp, self).__init__()
        self.opt = opt
        if opt[0]==0: # upsample+conv
            self.up = nn.Sequential(nn.ConvTranspose3d(in_num, in_num, cfg_up['pool_kernel'], cfg_up['pool_stride'], groups=in_num, bias=False),
                unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
            self.up._modules['0'].weight.data.fill_(1.0)
            # not supported yet: anisotropic upsample
            # self.up = nn.Sequential(nn.Upsample(scale_factor=cfg_up.pool_kernel, mode='nearest'),
            #    unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
        elif opt[0]==1: # group deconv, remember to initialize with (1,0)
            self.up = nn.Sequential(nn.ConvTranspose3d(in_num, in_num, cfg_up['pool_kernel'], cfg_up['pool_stride'], groups=in_num, bias=True),
                unitConv3dRBD(in_num, outUp_num, 1, 1, 0, '', True))
            self.up._modules['0'].weight.data.fill_(1.0)
        elif opt[0]==2: # residual deconv
            self.up = blockResNet(unitResBasic, 1, in_num, in_num, stride_size=cfg_down['pool_stride'], do_sample=-1, cfg=cfg_conv)
            outUp_num = in_num

        if opt[1]==0: # merge-crop
            self.mc = mergeCrop
            inConv_num = outUp_num+inLeft_num
        elif opt[1]==1: # merge-add
            self.mc = mergeAdd
            inConv_num = min(outUp_num,inLeft_num)

        if opt[2]==0: # deconv
            self.conv = blockVgg(2, inConv_num, outConv_num, cfg=cfg_conv)
        elif opt[2]==1: # residual
            self.conv = blockResNet(unitResBasic, 2, inConv_num, outConv_num, cfg=cfg_conv)

    def forward(self, x1, x2):
        # inputs1 from left-side (bigger)
        x2_up = self.up(x2)
        mc = self.mc(x1, x2_up)
        return self.conv(mc)

class unetCenter(nn.Module):
    def __init__(self, opt, in_num, out_num,
        cfg_conv={'pad_size':0, 'pad_type':'', 'has_bias':True, 'has_BN':False, 'relu_slope':0.005, 'has_dropout':0} ):
        super(unetCenter, self).__init__()
        self.opt = opt
        if opt[0]==0: # vgg
            self.conv = blockVgg(2, in_num, out_num, cfg=cfg_conv)
        elif opt[0]==1: # residual
            self.conv = blockResNet(unitResBasic, 2, in_num, out_num, cfg=cfg_conv)
    def forward(self, x):
        return self.conv(x)

class unetFinal(nn.Module):
    def __init__(self, opt, in_num, out_num):
        super(unetFinal, self).__init__()
        self.opt = opt
        if opt[0]==0: # vgg
            self.conv = unitConv3dRBD(in_num, out_num, 1, 1, 0, '', True)
        elif opt[0]==1: # resnet
            self.conv = blockResNet(unitResBasic, 1, in_num, out_num, 1, cfg={'pad_size':0, 'pad_type':'constant,0', 'has_BN':False, 'relu_slope':-1})

    def forward(self, x):
        return F.sigmoid(self.conv(x))


class unet3D(nn.Module): # symmetric unet
    # default global parameter 
    # opt_arch: change component 
    # component parameter
    def __init__(self, opt_arch=[[0,0],[0],[0,0,0],[0]], opt_param=[[0],[0],[0],[0]], 
                 in_num=1, out_num=3, filters=[24,72,216,648],
                 has_bias=True, has_BN=False,has_dropout=0,pad_size=0,pad_type='',relu_slope=0.005,
                 pool_kernel=(1,2,2), pool_stride=(1,2,2)):
        super(unet3D, self).__init__()
        self.depth = len(filters)-1 
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.relu_slope = relu_slope
        filters_in = [in_num] + filters[:-1]
        cfg_conv={'has_bias':has_bias,'has_BN':has_BN, 'has_dropout':has_dropout, 'pad_size':pad_size, 'pad_type':pad_type, 'relu_slope':relu_slope}
        cfg_pool={'pool_kernel':pool_kernel, 'pool_stride':pool_stride}

        # --- down arm ---
        self.down = nn.ModuleList([
                    unetDown(opt_arch[0], filters_in[x], filters_in[x+1], cfg_pool, cfg_conv, x) 
                    for x in range(len(filters)-1)]) 

        # --- center arm ---
        cfg_conv_c = copy.deepcopy(cfg_conv)
        if opt_param[1][0]==1: # pad for conv
            cfg_conv_c['pad_size']=1;cfg_conv_c['pad_type']='replicate';
        self.center = unetCenter(opt_arch[1], filters[-2], filters[-1], cfg_conv_c)

        # --- up arm ---
        cfg_conv_u = copy.deepcopy(cfg_conv)
        if opt_param[2][0]==1: # pad for conv
            cfg_conv_u['pad_size']=1;cfg_conv_c['pad_type']='replicate';
        self.up = nn.ModuleList([
                    unetUp(opt_arch[2], filters[x], filters[x-1], filters[x-1], filters[x-1], cfg_pool, cfg_conv_u, x)
                    for x in range(len(filters)-1,0,-1)])

        # --- final arm ---
        self.final = unetFinal(opt_arch[3], filters[0], out_num) 

    def forward(self, x):
        down_u = [None]*self.depth
        for i in range(self.depth):
            down_u[i], x = self.down[i](x)
        x = self.center(x)
        for i in range(self.depth):
             x = self.up[i](down_u[self.depth-1-i],x)
        return self.final(x)

class unet3D_PNI(nn.Module):
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_num=1, out_num=3, filters=[28,36,48,64,80], has_BN=True, relu_opt=0):
        super(unet3D_m2_v2, self).__init__()
        self.filters = filters 
        self.io_num = [in_num, out_num]
        self.res_num = len(filters)-2
        self.seq_num = (self.res_num+1)*2+1

        self.downC = nn.ModuleList(
                [unet_m2_conv([in_num], [1], [(1,5,5)], [(0,2,2)], [1], [False], [has_BN], [relu_opt])]
                + [unet_m2_BasicBlock(1, filters[0], False, has_BN, relu_opt)]
                + [unet_m2_BasicBlock(filters[x], filters[x+1], True, has_BN, relu_opt)
                      for x in range(1, self.res_num)]) 
        self.downS = nn.ModuleList(
                [nn.MaxPool3d((1,2,2), (1,2,2))
            for x in range(self.res_num+1)]) 
        self.center = unet_m2_BasicBlock(filters[-2], filters[-1], True, has_BN, relu_opt)
        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose3d(filters[self.res_num+1-x], filters[self.res_num+1-x], (1,2,2), (1,2,2), groups=filters[self.res_num+1-x], bias=False),
                nn.Conv3d(filters[self.res_num+1-x], filters[self.res_num-x], kernel_size=(1,1,1), stride=1, bias=True))
                for x in range(self.res_num+1)]) 
        # initialize upsample
        for x in range(self.res_num+1):
            self.upS[x]._modules['0'].weight.data.fill_(1.0)

        self.upC = nn.ModuleList( # same number of channels from the left
            [unet_m2_BasicBlock(filters[self.res_num-x], filters[self.res_num-x], True, has_BN, relu_opt)
                for x in range(self.res_num-1)]
            + [unet_m2_BasicBlock(filters[0], filters[0], False, has_BN, relu_opt)]
            + [nn.Conv3d(filters[0], out_num, kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=True)]) 

    def forward(self, x):
        down_u = [None]*(self.res_num+1)
        x = self.downC[0](x) # first 1x5x5
        for i in range(self.res_num+1):
            down_u[i] = self.downC[1+i](x)
            x = self.downS[i](down_u[i])
        x = self.center(x)
        for i in range(self.res_num+1):
            x = down_u[self.res_num-i] + self.upS[i](x)
            x = self.upC[i](x)
        x = self.upC[-1](x) # last 1x5x5
        return F.sigmoid(x)


