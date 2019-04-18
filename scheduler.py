import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithRestartsLR(_LRScheduler):
    '''Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Factor of increment of T_max each restart.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    '''

    def __init__(self, optimizer, T_max, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.save_flag = False
        super(CosineAnnealingWithRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.save_flag:
            self.save_flag = False

        # restart the scheduler
        # eta cannot be eta_min which is 0 by default
        if self.last_epoch == self.T_max:
            self.save_flag = True
            self.last_epoch = 0
            self.T_max *= self.T_mult
            return self.base_lrs

        # Otherwise use the same code as pytorch official implementation
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
