import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialBlock(nn.Module):
    def __init__(self, frame_dim, out_dim, kernal_size, stride):
        super(SpatialBlock, self).__init__()
        self.frame_dim = frame_dim
        self.out_dim = out_dim
        self.conv = nn.Conv2d(frame_dim, out_dim, kernel_size=kernal_size, stride=stride)

    def forward(self, x):
        x_re = x.view(-1, self.frame_dim, x.shape[2], x.shape[3])
        h = self.conv(x_re)
        h_out = h.view(x.shape[0], -1, h.shape[2], h.shape[3])
        return h_out

class TSMBlock(nn.Module):
    def __init__(self, frame_dim, out_dim, shift_dim, kernal_size, stride):
        super(TSMBlock, self).__init__()
        self.frame_dim = frame_dim
        self.out_dim = out_dim
        self.shift_dim = shift_dim
        self.conv = nn.Conv2d(frame_dim, out_dim, kernel_size=kernal_size, stride=stride)

    def forward(self, x):
        x_re = x.view(-1, self.frame_dim, x.shape[2], x.shape[3])
        h = self.conv(x_re)
        h_re = h.view(x.shape[0], -1, self.out_dim, h.shape[2], h.shape[3])
        zero_block = torch.zeros([x.shape[0], 1, self.shift_dim, h.shape[2], h.shape[3]])
        if x.get_device()>=0:
            zero_block = zero_block.to(x.get_device())
        h_shift1 = torch.cat([zero_block, h_re[:,1:,:self.shift_dim,:,:]], 1)
        h_shift2 = torch.cat([h_re[:,:-1,self.shift_dim:2*self.shift_dim,:,:], zero_block], 1)
        h_out = torch.cat([h_shift1, h_shift2, h_re[:,:,2*self.shift_dim:,:,:]], 2)
        h_out = h_out.view(x.shape[0], -1, h_out.shape[3], h_out.shape[4])
        return h_out
