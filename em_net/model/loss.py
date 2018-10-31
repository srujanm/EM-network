from __future__ import print_function, division
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch     
import numpy as np
# for malis
from em_segLib.seg_malis import malis_init, malis_loss_weights_both
from em_segLib.seg_util import mknhood3d


class MalisWeight:
    def __init__(self, conn_dims, opt_weight=0.5, opt_nb=1):
        # pre-compute 
        self.opt_weight = opt_weight
        if opt_nb == 1:
            self.nhood_data = mknhood3d(1).astype(np.int32).flatten()
        else:
            self.nhood_data = mknhood3d(1).astype(np.uint64).flatten()
        self.nhood_dims = np.array((3, 3), dtype=np.uint64)
        self.conn_dims = np.array(conn_dims[1:]).astype(np.uint64)  # dim=4
        self.pre_ve, self.pre_prodDims, self.pre_nHood = malis_init(self.conn_dims, self.nhood_data, self.nhood_dims)
        self.weight = np.zeros(conn_dims, dtype=np.float32)  # pre-allocate

    def get_weight(self, x_cpu, aff_cpu, seg_cpu):
        for i in range(x_cpu.shape[0]):
            self.weight[i] = malis_loss_weights_both(seg_cpu[i].flatten(), self.conn_dims, self.nhood_data,
                                                     self.nhood_dims, self.pre_ve, self.pre_prodDims,
                                                     self.pre_nHood, x_cpu[i].flatten(), aff_cpu[i].flatten(),
                                                     self.opt_weight).reshape(self.conn_dims)
        return self.weight[:x_cpu.shape[0]]


class DiceLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, smooth=100.0):
        super(DiceLoss, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.reduce = reduce

    def dice_loss(self, input_y, target):
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) /
                         ((iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input_y.size()[0])

    def dice_loss_batch(self, input_y, target):

        iflat = input_y.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        loss = 1 - ((2. * intersection + self.smooth) /
                    ((iflat**2).sum() + (tflat**2).sum() + self.smooth))
        return loss

    def forward(self, input_y, target):
        with torch.no_grad():
            if not (target.size() == input_y.size()):
                raise ValueError(
                    "Target size ({}) must be the same as input size ({})".format(target.size(), input_y.size()))

            if self.reduce:
                loss = self.dice_loss(input_y, target)
            else:
                loss = self.dice_loss_batch(input_y, target)
            return loss


class WeightedMSE(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(WeightedMSE, self).__init__(size_average, reduce)

    @staticmethod
    def weighted_mse_loss(input_y, target, weight):
        s1 = torch.prod(torch.tensor(input_y.size()[2:]).float())
        s2 = input_y.size()[0]
        norm_term = (s1 * s2).cuda()
        return torch.sum(weight * (input_y - target) ** 2) / norm_term

    def forward(self, input_y, target, weight):
        with torch.no_grad():
            return self.weighted_mse_loss(input_y, target, weight)


# define a customized loss function for future development
class WeightedBCELoss(_Loss):

    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss, self).__init__(size_average, reduce)

    def forward(self, input_y, target, weight):
        with torch.no_grad():
            return F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)


# Weighted binary cross entropy + Dice loss
class BCLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, smooth=10.0):
        super(BCLoss, self).__init__(size_average, reduce)
        self.smooth = smooth

    def dice_loss(self, input_y, target):
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input_y.size()[0])

    def forward(self, input_y, target, weight):
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        with torch.no_grad():
            loss1 = F.binary_cross_entropy(input_y, target, weight, self.size_average,
                                       self.reduce)
            loss2 = self.dice_loss(input_y, target)
            return loss1, loss2


# Focal Loss
class FocalLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, gamma=2):
        super(FocalLoss, self).__init__(size_average, reduce)
        self.gamma = gamma

    def focal_loss(self, input_y, target, weight):
        eps = 1e-7
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * ((1 - iflat) ** self.gamma)
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat ** self.gamma)
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input_y.size()[0])

    def forward(self, input_y, target, weight):
        """
        Weighted Focal Loss
        """
        with torch.no_grad():
            if not (target.size() == input_y.size()):
                raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(),
                                                                                               input_y.size()))

            loss = self.focal_loss(input_y, target, weight)
            return loss


# Focal Loss + Dice Loss
class BCLoss_focal(_Loss):
    def __init__(self, size_average=True, reduce=True, smooth=10.0, gamma=2):
        super(BCLoss_focal, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.gamma = gamma

    def dice_loss(self, input_y, target):
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) / ( (iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input_y.size()[0])

    def focal_loss(self, input_y, target, weight):
        eps = 1e-7
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            wflat = weight[index].view(-1)

            iflat = iflat.clamp(eps, 1.0 - eps)
            fc_loss_pos = -1 * tflat * torch.log(iflat) * (1 - iflat) ** self.gamma
            fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * iflat ** self.gamma
            fc_loss = fc_loss_pos + fc_loss_neg
            fc_loss = fc_loss * wflat # weighted focal loss

            loss += fc_loss.mean()
        
        return loss / float(input_y.size()[0])

    def forward(self, input_y, target, weight):
        """
        Weighted binary classification loss + Dice coefficient loss
        """
        with torch.no_grad():
            if not (target.size() == input_y.size()):
                raise ValueError(
                    "Target size ({}) must be the same as input size ({})".format(target.size(), input_y.size()))

            loss1 = self.focal_loss(input_y, target, weight)
            loss2 = self.dice_loss(input_y, target)
            return loss1, loss2


# Focal Loss
class FocalLossMul(_Loss):
    def __init__(self, size_average=True, reduce=True, gamma=2):
        super(FocalLossMul, self).__init__(size_average, reduce)
        self.gamma = gamma

    def focal_loss(self, input_y, target, weight):
        eps = 1e-6
        loss = 0.

        for index in range(input_y.size()[0]):
            sample_loss = 0.
            for channel in range(input_y.size()[1]):
                iflat = input_y[index, channel].view(-1)
                tflat = target[index, channel].view(-1)
                wflat = weight[index].view(-1) # use the same weight matrix for all channels 

                iflat = iflat.clamp(eps, 1.0 - eps)
                fc_loss_pos = -1 * tflat * torch.log(iflat) * ((1 - iflat) ** self.gamma)
                fc_loss_neg = -1 * (1-tflat) * torch.log(1 - iflat) * (iflat ** self.gamma)
                fc_loss = fc_loss_pos + fc_loss_neg
                fc_loss = fc_loss * wflat # weighted focal loss

                sample_loss += fc_loss.mean()

            loss += sample_loss
        
        return loss / float(input_y.size()[0])

    def forward(self, input_y, target, weight):
        """
        Weighted Focal Loss
        """
        with torch.no_grad():
            if not (target.size() == input_y.size()):
                raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(),
                                                                                               input_y.size()))

            loss = self.focal_loss(input_y, target, weight)
            return loss
