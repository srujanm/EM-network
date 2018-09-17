import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm3d

def conv3d_pad(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
               dilation=(1,1,1), padding=(1,1,1), bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return  nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias))     

def conv3d_bn_non(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              

def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, 
                       dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))                                   

# -- 0.2 u-net++ --
# u-net with dilated convolution, ELU and synchronized BN

class unet_SE_synBN(nn.Module):
    # unet architecture with residual blocks
    def __init__(self, in_num=1, out_num=1, filters=[64,96,128,256], aniso_num=2):
        super(unet_SE_synBN, self).__init__()
        self.filters = filters 
        self.layer_num = len(filters) # 4
        self.aniso_num = aniso_num # the number of anisotropic conv layers

        self.downC = nn.ModuleList(
                  [res_unet_AnisoBlock_dilation(in_num, filters[0])]
                + [res_unet_AnisoBlock_dilation(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1)] 
                + [res_unet_IsoBlock(filters[x], filters[x+1])
                      for x in range(self.aniso_num-1, self.layer_num-2)]
                      ) 

        self.downS = nn.ModuleList(
                [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
                    for x in range(self.aniso_num)]
              + [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
                    for x in range(self.aniso_num, self.layer_num-1)]
                )

        self.center = res_unet_IsoBlock(filters[-2], filters[-1])

        self.upS = nn.ModuleList(
            [nn.Sequential(
                nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False),
                conv3d_bn_non(filters[self.layer_num-1-x], filters[self.layer_num-2-x], 
                              kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(self.layer_num-self.aniso_num-1)]
          + [nn.Sequential(
                nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
                conv3d_bn_non(filters[self.layer_num-1-x], filters[self.layer_num-2-x], 
                              kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True))
                for x in range(1, self.aniso_num+1)]
            )

        self.upC = nn.ModuleList(
            [res_unet_IsoBlock(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(self.layer_num-self.aniso_num-1)]
          + [res_unet_AnisoBlock_dilation(filters[self.layer_num-2-x], filters[self.layer_num-2-x])
                for x in range(1, self.aniso_num)]
          + [res_unet_AnisoBlock_dilation(filters[0], filters[0])]
            )

        self.fconv = conv3d_bn_non(filters[0], out_num, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=True)
        self.elu = nn.ELU(inplace=True)
        self.softmax = nn.Softmax(dim=1) 
        self.sigmoid = nn.Sigmoid()     

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

    def forward(self, x):
        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        for i in range(self.layer_num-1):
            x = down_u[self.layer_num-2-i] + self.upS[i](x)
            x = self.elu(x)
            x = self.upC[i](x)
        # convert to probability  
        x = self.fconv(x)  
        x = self.sigmoid(x)
        return x

# -- 0.3 building blocks--
class res_unet_IsoBlock(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_IsoBlock, self).__init__()
        self.block1 = conv3d_bn_elu(in_planes,  out_planes, kernel_size=(3,3,3), 
                                    stride=1, padding=(1,1,1), bias=False)
        self.block2 = nn.Sequential(
            conv3d_bn_elu(out_planes,  out_planes, kernel_size=(3,3,3), 
                stride=1, padding=(1,1,1), bias=False),
            conv3d_bn_non(out_planes,  out_planes, kernel_size=(3,3,3), 
                stride=1, padding=(1,1,1), bias=False))
        self.block3 = nn.ELU(inplace=True)    

    def forward(self, x):
        residual  = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out 

class res_unet_AnisoBlock_dilation(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(res_unet_AnisoBlock_dilation, self).__init__() 
        self.se_layer = SELayer(channel=out_planes, reduction=4)
        self.se_layer_sc = SELayer_cs(channel=out_planes, reduction=4)

        self.inconv = conv3d_bn_elu(in_planes,  out_planes, kernel_size=(3,3,3), 
                                    stride=1, padding=(1,1,1), bias=True)

        self.block1 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,1,1), padding=(0,1,1), bias=False)
        self.block2 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False)
        self.block3 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False)
        self.block4 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False)                                                                                  

        self.activation = nn.ELU(inplace=True)    

    def forward(self, x):
        residual  = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(F.elu(x1, inplace=True))
        x3 = self.block3(F.elu(x2, inplace=True))
        x4 = self.block4(F.elu(x3, inplace=True))

        out = residual + x1 + x2 + x3 + x4
        out = self.se_layer_sc(out)
        out = self.activation(out)
        return out 

# Squeeze-and-Excitation Layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class SELayer_cs(nn.Module):
    # Squeeze-and-excitation layer (channel & spatial)
    def __init__(self, channel, reduction=4):
        super(SELayer_cs, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

        self.sc = nn.Sequential(
                nn.Conv3d(channel, 1, kernel_size=(1,1,1)),
                SynchronizedBatchNorm3d(1),
                nn.ELU(inplace=True),
                nn.MaxPool3d(kernel_size=(1,8,8), stride=(1,8,8)),
                conv3d_bn_elu(1, 1, kernel_size=(3,3,3), padding=(1,1,1)),
                nn.Upsample(scale_factor=(1,8,8), mode='trilinear', align_corners=False),
                nn.Conv3d(1, channel, kernel_size=(1,1,1)),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid())     

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        z = self.sc(x)
        return (x * y) + (x * z)    


# model for visualization purpose
class unet_SE_synBN_visualization(unet_SE_synBN):
    def __init__(self, in_num=1, out_num=3, filters=[64,96,128,256], aniso_num=2):
        super(unet_SE_synBN_visualization, self).__init__(in_num, 
                                      out_num, filters, aniso_num)  

    def forward(self, x):

        down_u = [None]*(self.layer_num-1)
        for i in range(self.layer_num-1):
            down_u[i] = self.downC[i](x)
            x = self.downS[i](down_u[i])

        x = self.center(x)

        output_up0 = [None]*(self.layer_num-1)
        output_up1 = [None]*(self.layer_num-1)

        for i in range(self.layer_num-1):
            output_up0[i] = self.upS[i](x)
            x = down_u[self.layer_num-2-i] + output_up0[i]
            x = self.elu(x)
            x = self.upC[i](x)
            output_up1[i] = x
        # convert to probability
        x = self.fconv(x)    
        x = self.sigmoid(x)
        return x, down_u, output_up0, output_up1