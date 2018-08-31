from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset, normalize, transform
from pointnet import PointNetCls
import torch.nn.functional as F

from tqdm import tqdm

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(torch.from_numpy(np.array(target)))
    return [data, target]

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

parser.add_argument('--base_features', type=int, default=32, help='base feature size in network (pointnet uses 64)')


opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'


print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)




dataset = PartDataset(root = '../PointNet_Data', classification = True, npoints = opt.num_points, transform = normalize)

# dataset = PartDataset(root = '../PointNet_Data', classification = True, npoints = opt.num_points,
#     transform = transform(translation_range=(-0.0, 0.0), scale_range=(0.8, 1.25), rotation_range=(-10, 10)))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers))#, collate_fn=my_collate)

test_dataset = PartDataset(root = '../PointNet_Data', classification = True, train = False, npoints = opt.num_points, transform = normalize)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers) )#,collate_fn=my_collate)

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(n = opt.base_features, k = num_classes)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

device = torch.cuda.current_device()
#optimizer = optim.SGD(classifier.parameters(), lr=opt.lr, momentum=0.9)
optimizer = optim.Adam(classifier.parameters(), lr=opt.lr)

classifier.to(device)
classifier.eval()

num_batch = len(dataset)/opt.batchSize


def prepare(data):
    points, target = data
    return points.transpose(2,1).to(device), target[:,0].to(device)

for epoch in range(opt.nepoch):

    correct = 0
    total_loss = 0
    count = 0

    for data in tqdm(dataloader):
        classifier.train()

        points, target = prepare(data)
        optimizer.zero_grad()
        pred, _ = classifier(points)

        loss = F.nll_loss(pred, target, reduction='sum')
        # print(pred.exp(), target, loss.item())

        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]

        correct += pred_choice.eq(target.data).sum().item()
        total_loss += loss.item()
        count += points.size(0)

    print('[%d] %s loss: %f accuracy: %f' %(epoch, 'train', total_loss / count, correct / count))

    correct = 0
    total_loss = 0
    count = 0

    with torch.no_grad():
        classifier.eval()
        for data in tqdm(testdataloader):

            points, target = prepare(data)
            # points, target = points.cuda(), target.cuda()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target, reduction='sum')
            pred_choice = pred.data.max(1)[1]

            correct += pred_choice.eq(target.data).sum().item()

            total_loss += loss.item()
            count += points.size(0)

        print('[%d] %s loss: %f accuracy: %f' %(epoch, blue('test'), total_loss  / count, correct / count))


    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
