#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import argparse
from os import listdir
from PIL import Image
import numpy as np

from unet import UNet
from myloss import dice_coeff
from script import resize, getBoundingBox, applyMask, cropBlack

log_frequency = 100

# Parse Arguments
parser = argparse.ArgumentParser(description="Trains the unet")
parser.add_argument("data", type=str, help="Path to a folder containing data to train")
parser.add_argument("truth", type=str, help="Path to a folder containing the ground truth to train with")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate used to optimize")
parser.add_argument("-m", dest="momentum", type=float, default=0.5, help="Momentum used by the optimizer")
parser.add_argument("-e", dest="epoch", type=int, default=5, help="Number of training epochs")
parser.add_argument("-b", dest="batch_size", type=int, default=4, help="Size of the batches used in training")
parser.add_argument("--cpu", dest="cpu", action="store_true", required=False, help="Use CPU instead of CUDA.")
args = parser.parse_args()

use_cuda = args.cpu or torch.cuda.is_available()

data_names = listdir(args.data)
data_names.sort()

truth_names = listdir(args.truth)
truth_names.sort()

if len(data_names) != len(truth_names):
    print("Need the same amount of data and truth")
    exit(-1)

model = UNet(3,1)
if use_cuda:
    model.cuda()
    print("Using CUDA")

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.BCELoss()

def train(epoch):
    model.train()

    # from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for batch_idx, batch_range in enumerate(batch(range(0, len(data_names)), args.batch_size)):
        images, masks = [], []
        for item_idx in batch_range:
            image = Image.open(data_names[item_idx])
            mask = Image.open(truth_names[item_idx])
            image, mask = resize(image, 2), resize(mask, 2)
            image = applyMask(image, getBoundingBox(mask, 20))
            image, mask = cropBlack(image, mask)

            images.append(np.array(image))
            masks.append(np.array(mask))

        batch_data, batch_truth = torch.FloatTensor(np.array(images)), torch.ByteTensor(np.array(masks))

        if use_cuda:
            batch_data, batch_truth = batch_data.cuda(), batch_truth.cuda()
        data, truth = Variable(batch_data), Variable(batch_truth)

        optimizer.zero_grad()

        output = model(data)
        output_probs = F.sigmoid(output).view(-1)

        loss = criterion(output_probs, truth.view(-1).float())
        loss.backward()

        optimizer.step()

        if batch_idx % log_frequency == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_names),
                100. * batch_idx / len(data_names), loss.data[0]))

def test():
    model.test()
    tot_dice = 0
    for i, (data_name, truth_name) in enumerate(zip(data_names, truth_names)):
        image = Image.open(data_name)
        mask = Image.open(truth_name)
        image, mask = resize(image, 2), resize(mask, 2)
        image = applyMask(image, getBoundingBox(mask, 20))
        image, mask = cropBlack(image, mask)

        iter_data, iter_truth = torch.FloatTensor(np.array(image)), torch.ByteTensor(np.array(mask))
        if use_cuda:
            iter_data, iter_truth = iter_data.cuda(), iter_truth.cuda()
        data, truth = Variable(iter_data, volatile=True), Variable(iter_truth, volatile=True)

        output = model(data)
        output_probs = (F.sigmoid(output) > 0.6).float()

        dice = dice_coeff(output_probs, truth.float()).data[0]
        tot_dice += dice

    print('Validation Dice Coeff: {}'.format(tot_dice / len(data_names)))


for epoch in range(1, args.epoch + 1):
    train(epoch)
    test()
