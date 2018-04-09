#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import argparse
from os import listdir
import os.path
from PIL import Image
import numpy as np

from unet import UNet
from myloss import dice_coeff
from script import resize, getBoundingBox, applyMask, cropBlack

log_frequency = 1

# Parse Arguments
parser = argparse.ArgumentParser(description="Trains the unet")
parser.add_argument("data", type=str, help="Path to a folder containing data to train")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate used to optimize")
parser.add_argument("-m", dest="momentum", type=float, default=0.5, help="Momentum used by the optimizer")
parser.add_argument("-e", dest="epoch", type=int, default=5, help="Number of training epochs")
parser.add_argument("-b", dest="batch_size", type=int, default=4, help="Size of the batches used in training")
parser.add_argument("--cpu", dest="cpu", action="store_true", required=False, help="Use CPU instead of CUDA.")
args = parser.parse_args()

use_cuda = args.cpu or torch.cuda.is_available()

input_path = os.path.join(args.data, "input")
trimap_path = os.path.join(args.data, "trimap")
target_path = os.path.join(args.data, "target")

ids = [os.path.splitext(filename)[0].split('_') for filename in listdir(input_path)]
np.random.shuffle(ids)

model = UNet(4,4)
if use_cuda:
    model.cuda()
    print("Using CUDA")

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.MSELoss()

def train(epoch):
    model.train()

    # from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for batch_idx, batch_range in enumerate(batch(ids, args.batch_size)):
        images, targets = [], []
        for i, j in batch_range:
            input_filename = os.path.join(input_path, str(i) + '_' + str(j) + '.jpg')
            trimap_filename = os.path.join(trimap_path, str(i) + '_trimap.jpg')
            target_filename = os.path.join(target_path, str(i) + '.png')
            print(input_filename)
            with Image.open(input_filename) as image, \
                 Image.open(trimap_filename) as trimap, \
                 Image.open(target_filename) as target:

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = np.array(image).transpose((2, 0, 1))
                trimap = np.array(trimap)[np.newaxis, ...]
                print(image.shape, trimap.shape)
                image = np.concatenate((image, trimap), axis = 0)
                target = np.array(target, ndmin=3) / 255

                images.append(image)
                targets.append(target)

        batch_input = torch.FloatTensor(np.array(images))
        batch_target = torch.FloatTensor(np.array(targets))

        if use_cuda:
            batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
        batch_input, batch_target = Variable(batch_input), Variable(batch_target)

        optimizer.zero_grad()

        output = model(batch_input)
        predicted = F.sigmoid(output).view(-1)
        loss = criterion(predicted, batch_target.view(-1))
        loss.backward()
        optimizer.step()

        if batch_idx % log_frequency == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(batch_input), len(ids),
                100. * (batch_idx+1) * len(batch_input) / len(ids), loss.data[0]))

def test():
    model.eval()
    tot_dice = 0
    for i, (data_name, truth_name) in enumerate(zip(data_names, truth_names)):
        image = Image.open(os.path.join(args.data, data_name))
        mask = Image.open(os.path.join(args.truth, truth_name))
        image, mask = resize(image, 2), resize(mask, 2)
        image = applyMask(image, getBoundingBox(mask, 20))
        #image, mask = cropBlack(image, mask)

        iter_data, iter_truth = torch.FloatTensor(np.array(image, ndmin=4).transpose(0,3,1,2)), torch.ByteTensor(np.array(mask, ndmin=4))
        if use_cuda:
            iter_data, iter_truth = iter_data.cuda(), iter_truth.cuda()

        image.close()
        mask.close()

        data, truth = Variable(iter_data, volatile=True), Variable(iter_truth, volatile=True)

        output = model(data)
        output_probs = (F.sigmoid(output) > 0.6).float()

        dice = dice_coeff(output_probs, truth.float()).data[0]
        tot_dice += dice

    print('Validation Dice Coeff: {}'.format(tot_dice / len(data_names)))


for epoch in range(1, args.epoch + 1):
    train(epoch)
    #test()
