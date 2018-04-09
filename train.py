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

train_data_update_freq = 1
test_data_update_freq = 50

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
output_path = os.path.join(args.data, "output")

ids = [os.path.splitext(filename)[0].split('_') for filename in listdir(input_path)]
np.random.shuffle(ids)
split_point = int(round(0.99*len(ids))) #using 70% as training and 30% as Validation
train_ids = ids[0:split_point]
valid_ids = ids[split_point:len(ids)]


model = UNet(4,4)
#if use_cuda:
#    model.cuda()
#    print("Using CUDA")

optimizer = tf.train.GradientDescentOptimizer(args.lr)
#criterion = nn.MSELoss()

input_images = tf.placeholder(tf.float32, shape=[None, 240, 180, 4])
target_images = tf.placeholder(tf.float32, shape=[None, 240, 180, 4])

output = tf.sigmoid(tf.squeeze(model(input_images)))
loss = tf.losses.mean_squared_error(target_images, output)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def test():
    total_loss = 0
    #for i, (data_name, truth_name) in enumerate(zip(data_names, truth_names)):
    for (i, j) in valid_ids:
        input_filename = os.path.join(input_path, str(i) + '_' + str(j) + '.jpg')
        trimap_filename = os.path.join(trimap_path, str(i) + '_trimap.jpg')
        target_filename = os.path.join(target_path, str(i) + '.png')
        print(input_filename)
        with Image.open(input_filename) as image, \
             Image.open(trimap_filename) as trimap, \
             Image.open(target_filename) as target:
            image = resize(image, 4)
            trimap = resize(trimap, 4)
            target = resize(target, 4)

            image = np.array(image)
            trimap = np.array(trimap)[..., np.newaxis]
            image = np.concatenate((image, trimap), axis = 2)

            target = np.array(target) / 255

            l, o = sess.run([loss, output], feed_dict={
                input_images: image[np.newaxis, ...],
                target_images: target[np.newaxis, ...],
                })
            total_loss += l

            o = Image.fromarray((o * 255).astype(np.uint8))
            o.save(os.path.join(output_path, str(i) + '.png'))


    print('Validation Loss: {}'.format(total_loss / len(valid_ids)))


def train(epoch):

    # from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    for batch_idx, batch_range in enumerate(batch(train_ids, args.batch_size)):
        images, targets = [], []
        for i, j in batch_range:
            input_filename = os.path.join(input_path, str(i) + '_' + str(j) + '.jpg')
            trimap_filename = os.path.join(trimap_path, str(i) + '_trimap.jpg')
            target_filename = os.path.join(target_path, str(i) + '.png')
            print(input_filename)
            with Image.open(input_filename) as image, \
                 Image.open(trimap_filename) as trimap, \
                 Image.open(target_filename) as target:

                image = resize(image, 4)
                trimap = resize(trimap, 4)
                target = resize(target, 4)

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

        if batch_idx % train_data_update_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * args.batch_size, len(ids),
                100. * (batch_idx+1) * args.batch_size / len(ids), l))

        if batch_idx % test_data_update_freq == 0:
            test()


for epoch in range(1, args.epoch + 1):
    train(epoch)
