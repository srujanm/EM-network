import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from em_net.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm3d


# -- 0. util layers--
def merge_crop(x1, x2):
    # x1 bigger
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2, x1.dim())]
    return torch.cat([x2, x1[:, :, offset[0]:offset[0]+x2.size(2),
                          offset[1]:offset[1]+x2.size(3), offset[2]:offset[2]+x2.size(4)]], 1)


def merge_add(x1, x2):
    # x1 bigger
    offset = [(x1.size()[x]-x2.size()[x])//2 for x in range(2, x1.dim())]
    return x2 + x1[:, :, offset[0]:offset[0]+x2.size(2), offset[1]:offset[1]+x2.size(3), offset[2]:offset[2]+x2.size(4)]


# -- 1 conv blocks--
def unet_m2_conv(in_num, out_num, kernel_size, pad_size, stride_size, has_bias, batch_norm, relu_opt):
    layers = []
    for i in range(len(in_num)):
        layers.append(
            nn.Conv3d(in_num[i], out_num[i], kernel_size=kernel_size[i], padding=pad_size[i], stride=stride_size[i],
                      bias=has_bias[i]))
        if batch_norm[i]:
            layers.append(nn.BatchNorm3d(out_num[i]))
        if relu_opt[i] == 0:
            layers.append(nn.ReLU(inplace=True))
        elif relu_opt[i] == 1:
            layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


def conv3d_pad(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
               bias=False):
    # the size of the padding should be a 6-tuple        
    padding = tuple([x for x in padding for _ in range(2)][::-1])
    return nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                          stride=stride, padding=0, dilation=dilation, bias=bias))


def conv3d_bn_non(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
                  bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes))              


def conv3d_bn_elu(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1),
                  bias=False):
    return nn.Sequential(
            conv3d_pad(in_planes, out_planes, kernel_size, stride, dilation, padding, bias),
            SynchronizedBatchNorm3d(out_planes),
            nn.ELU(inplace=True))                                   


# -- 2. residual blocks--
class ResUNetIsoBlock(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(ResUNetIsoBlock, self).__init__()
        self.block1 = conv3d_bn_elu(in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
                                    bias=False)
        self.block2 = nn.Sequential(
            conv3d_bn_elu(out_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False),
            conv3d_bn_non(out_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False))
        self.block3 = nn.ELU(inplace=True)    

    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        return out 


class ResUNetAnisoBlockDilation(nn.Module):
    # Basic residual module of unet
    def __init__(self, in_planes, out_planes):
        super(ResUNetAnisoBlockDilation, self).__init__()
        self.se_layer = SELayer(channel=out_planes, reduction=4)
        self.se_layer_sc = SELayerCS(channel=out_planes, reduction=4)

        self.inconv = conv3d_bn_elu(in_planes,  out_planes, kernel_size=(3, 3, 3),
                                    stride=1, padding=(1, 1, 1), bias=True)

        self.block1 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1, 3, 3),
                                    stride=1, dilation=(1,1,1), padding=(0, 1, 1), bias=False)
        self.block2 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,2,2), padding=(0,2,2), bias=False)
        self.block3 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,4,4), padding=(0,4,4), bias=False)
        self.block4 = conv3d_bn_non(out_planes,  out_planes, kernel_size=(1,3,3), 
                                    stride=1, dilation=(1,8,8), padding=(0,8,8), bias=False)                                                                                  

        self.activation = nn.ELU(inplace=True)    

    def forward(self, x):
        residual = self.inconv(x)

        x1 = self.block1(residual)
        x2 = self.block2(F.elu(x1, inplace=True))
        x3 = self.block3(F.elu(x2, inplace=True))
        x4 = self.block4(F.elu(x3, inplace=True))

        out = residual + x1 + x2 + x3 + x4
        out = self.se_layer_sc(out)
        out = self.activation(out)
        return out 


# -- 3. squeeze-and-excitation layer --
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


class SELayerCS(nn.Module):
    # Squeeze-and-excitation layer (channel & spatial)
    def __init__(self, channel, reduction=4):
        super(SELayerCS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

        self.sc = nn.Sequential(
                nn.Conv3d(channel, 1, kernel_size=(1, 1, 1)),
                SynchronizedBatchNorm3d(1),
                nn.ELU(inplace=True),
                nn.MaxPool3d(kernel_size=(1, 8, 8), stride=(1, 8, 8)),
                conv3d_bn_elu(1, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear', align_corners=False),
                nn.Conv3d(1, channel, kernel_size=(1, 1, 1)),
                SynchronizedBatchNorm3d(channel),
                nn.Sigmoid())     

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        z = self.sc(x)
        return (x * y) + (x * z)    
