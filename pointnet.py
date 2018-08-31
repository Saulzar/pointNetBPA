from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


def normalization(n):
    return nn.BatchNorm1d(n)
#    return nn.GroupNorm(num_groups = int(n/16), num_channels=n)


    # def id(x):
    #     return x
    # return id


class STN3d(nn.Module):
    def __init__(self, n):
        super(STN3d, self).__init__()
        self.n = n

        self.conv1 = torch.nn.Conv1d(3, n, 1)
        self.conv2 = torch.nn.Conv1d(n, n*2, 1)
        self.conv3 = torch.nn.Conv1d(n*2, n*16, 1)
        self.mp1 = torch.nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(n*16, n*8)
        self.fc2 = nn.Linear(n*8, n*4)
        self.fc3 = nn.Linear(n*4, 9)
        self.relu = nn.ReLU()

        self.bn1 = normalization(n)
        self.bn2 = normalization(n*2)
        self.bn3 = normalization(n*16)
        self.bn4 = normalization(n*8)
        self.bn5 = normalization(n*4)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, self.n*16)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, n, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.n = n

        self.stn = STN3d(n)
        self.conv1 = torch.nn.Conv1d(3, n, 1)
        self.conv2 = torch.nn.Conv1d(n, n*2, 1)
        self.conv3 = torch.nn.Conv1d(n*2, n*16, 1)
        self.bn1 = normalization(n)
        self.bn2 = normalization(n*2)
        self.bn3 = normalization(n*16)
        self.mp1 = torch.nn.AdaptiveMaxPool1d(1)
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, self.n*16)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, n*16, 1) #.repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self, n=16, k = 2):
        super(PointNetCls, self).__init__()
        self.n = n
        self.feat = PointNetfeat(n, global_feat=True)
        self.fc1 = nn.Linear(n*16, n*8)
        self.fc2 = nn.Linear(n*8, n*4)
        self.fc3 = nn.Linear(n*4, k)
        self.bn1 = normalization(n*8)
        self.bn2 = normalization(n*4)
        self.relu = nn.ReLU()


    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    # seg = PointNetDenseCls(k = 3)
    # out, _ = seg(sim_data)
    # print('seg', out.size())
