import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        # self.cat = {}
        self.categories = ['cast-off', 'impact', 'expirated']

        self.classification = classification

        # with open(self.catfile, 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.cat[ls[0]] = ls[1]
        #print(self.cat)
        # if not class_choice is  None:
        #     self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for category in self.categories:
            #print('category', item)
            self.meta[category] = []
            dir_point = os.path.join(self.root, category)
            # dir_seg = os.path.join(self.root, self.cat[category], 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[category].append((os.path.join(dir_point, token + '.pts')))

        self.datapath = []
        for category in self.categories:
            for fn in self.meta[category]:
                self.datapath.append((category, fn))


        self.classes = dict(zip(sorted(self.categories), range(len(self.categories))))
        self.num_seg_classes = 0
        # if not self.classification:
        #     for i in range(len(self.datapath)//50):
        #         l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
        #         if l > self.num_seg_classes:
        #             self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        # seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        # point_set = point_set[choice, :]
        # seg = seg[choice]
        point_set = torch.from_numpy(point_set)        
        # seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = '/media/cba62/Elements/PointNet_Data',  classification = True)
    print(len(d))
    ps, cls = d[1]
    print(ps.size(), ps.type(), cls.size(), cls.type())

    # d = PartDataset(root = '../shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.size(), ps.type(), cls.size(),cls.type())
