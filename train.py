#!/usr/bin/python3

import tensorflow as tf

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

input_path = os.path.join(args.data, "input")
trimap_path = os.path.join(args.data, "trimap")
target_path = os.path.join(args.data, "target")

ids = [os.path.splitext(filename)[0].split('_') for filename in listdir(input_path)]
np.random.shuffle(ids)

model = UNet(4,4)
#if use_cuda:
#    model.cuda()
#    print("Using CUDA")

optimizer = tf.train.GradientDescentOptimizer(args.lr)
#criterion = nn.MSELoss()

def train(epoch):

    # from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    input_images = tf.placeholder(tf.float32, shape=[args.batch_size, 960, 720, 4])
    target_images = tf.placeholder(tf.float32, shape=[args.batch_size, 960, 720, 4])

    output = tf.sigmoid(tf.squeeze(model(input_images)))
    loss = tf.losses.mean_squared_error(target_images, output)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

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

                image = np.array(image)
                trimap = np.array(trimap)[..., np.newaxis]

                image = np.concatenate((image, trimap), axis = 2)
                target = np.array(target) / 255

                images.append(image)
                targets.append(target)

        _, l = sess.run([optimizer.minimize(loss), loss], feed_dict={
            input_images: np.array(images),
            target_images: np.array(targets),
            })

        if batch_idx % log_frequency == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * args.batch_size, len(ids),
                100. * (batch_idx+1) * args.batch_size / len(ids), l))

"""def test():
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

    print('Validation Dice Coeff: {}'.format(tot_dice / len(data_names)))"""


for epoch in range(1, args.epoch + 1):
    train(epoch)
    #test()
