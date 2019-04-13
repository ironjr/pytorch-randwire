import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SeparableConv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, groups=in_planes)
        self.pointwise = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Node(nn.Module):
    def __init__(self, in_planes, planes, fin, downsample=False):
        super(Node, self).__init__()
        stride = 2 if downsample else 1

        # Top layer receives input from aggregation node (one)
        # NOTE: Top layer nodes require input weight
        if fin == 0:
            fin = 1
        self.w = nn.Parameter(torch.randn((1, fin))) # TODO Initialization?
        self.conv = SeparableConv(in_planes, planes, stride=stride)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        '''

        Args:
            x: A Tensor with fanin dimension in size (N, C, H, W, F)

        Returns:
            Single Tensor with size (N, C, H, W)
        '''
        x = F.linear(x, self.w).squeeze(-1) # (N,Cin,H,W)
        out = self.bn(self.conv(F.relu(x)))
        return out

