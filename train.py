#!/usr/bin/python3

import argparse
import os
import random
import logging

import tensorflow as tf
import numpy as np
from PIL import Image

from unet import UNet, Discriminator
from script import resize

model_name = "matting"

logging.basicConfig(level=logging.INFO)

# Parse Arguments
parser = argparse.ArgumentParser(description="Trains the unet")
parser.add_argument("data", type=str,
    help="Path to a folder containing data to train")
parser.add_argument("--lr", type=float, default=1.0,
    help="Learning rate used to optimize")
parser.add_argument("--d_coeff", type=float, default=1.0,
    help="Discriminator loss coefficient")
parser.add_argument("--nb_epoch", dest="nb_epoch", type=int, default=5,
    help="Number of training epochs")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=8,
    help="Size of the batches used in training")
parser.add_argument('--checkpoint', type=int, default=None,
    help='Saved session checkpoint, -1 for latest.')
parser.add_argument('--logdir', default="log/" + model_name,
    help='Directory where logs should be written.')
args = parser.parse_args()

input_path = os.path.join(args.data, "input")
trimap_path = os.path.join(args.data, "trimap")
target_path = os.path.join(args.data, "target")
output_path = os.path.join(args.data, "output")

train_data_update_freq = args.batch_size
test_data_update_freq = 50*args.batch_size
sess_save_freq = 100*args.batch_size

if not os.path.isdir(output_path):
    os.makedirs(output_path)

if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

ids = [[int(i) for i in os.path.splitext(filename)[0].split('_')] for filename in os.listdir(input_path)]
np.random.shuffle(ids)
split_point = int(round(0.85*len(ids))) #using 70% as training and 30% as Validation
train_ids = tf.get_variable('train_ids', initializer=ids[0:split_point], trainable=False)
valid_ids = tf.get_variable('valid_ids', initializer=ids[split_point:len(ids)], trainable=False)

global_step = tf.get_variable('global_step', initializer=0, trainable=False)

n_iter = int(args.nb_epoch * int(train_ids.shape[0]))

def apply_trimap(images, output, alpha):
    masked_output = []
    for channel in range(4):
        masked_output.append(output[:,:,:,channel])
        masked_output[channel] = tf.where(alpha < 0.25, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = tf.where(alpha > 0.75, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = masked_output[channel]
    masked_output = tf.stack(masked_output, 3)
    return masked_output

input_images = tf.placeholder(tf.float32, shape=[None, 240, 180, 4])
target_images = tf.placeholder(tf.float32, shape=[None, 240, 180, 4])
alpha = target_images[:,:,:,3][..., np.newaxis]

with tf.variable_scope("Gen"):
    gen = UNet(4,1)
    output = tf.sigmoid(gen(input_images))
    g_loss = tf.losses.mean_squared_error(alpha, output)
with tf.variable_scope("Disc"):
    disc = Discriminator(1)
    d_real = disc(alpha)
    d_fake = disc(output)
    d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))

a_loss = g_loss + args.d_coeff * d_loss

g_loss_summary = tf.summary.scalar("g_loss", g_loss)
d_loss_summary = tf.summary.scalar("d_loss", d_loss)
a_loss_summary = tf.summary.scalar("a_loss", a_loss)

summary_op = tf.summary.merge(
    [g_loss_summary, d_loss_summary, a_loss_summary])

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')

g_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(g_loss, global_step=global_step, var_list=g_vars)
a_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(a_loss, global_step=global_step, var_list=g_vars)
d_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(-d_loss, global_step=global_step, var_list=d_vars)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_writer = tf.summary.FileWriter(args.logdir + '/train')
test_writer = tf.summary.FileWriter(args.logdir + '/test')
saver = tf.train.Saver()
if args.checkpoint is not None and os.path.exists(os.path.join(args.logdir, 'checkpoint')):
    if args.checkpoint == -1:#latest checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(args.logdir))
    else:#Specified checkpoint
        saver.restore(sess, os.path.join(args.logdir, model_name+".ckpt-"+str(args.checkpoint)))
    logging.debug('Model restored to step ' + str(global_step.eval(sess)))


train_ids = list(train_ids.eval(sess))
valid_ids = list(valid_ids.eval(sess))

# from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def load_batch(batch_ids):
    images, targets = [], []
    for i, j in batch_ids:
        input_filename = os.path.join(input_path, str(i) + '_' + str(j) + '.jpg')
        trimap_filename = os.path.join(trimap_path, str(i) + '_trimap.jpg')
        target_filename = os.path.join(target_path, str(i) + '.png')
        logging.debug(input_filename)
        logging.debug(trimap_filename)
        logging.debug(target_filename)
        image = resize(Image.open(input_filename), 4)
        trimap = resize(Image.open(trimap_filename), 4)
        target = resize(Image.open(target_filename), 4)

        image = np.array(image)
        trimap = np.array(trimap)[..., np.newaxis]
        image = np.concatenate((image, trimap), axis = 2).astype(np.float32) / 255

        target = np.array(target).astype(np.float32) / 255

        images.append(image)
        targets.append(target)

    return np.asarray(images), np.asarray(targets)


def test_step(batch_idx):
    batch_range = random.sample(train_ids, args.batch_size)

    images, targets = load_batch(batch_range)

    loss, demo, summary = sess.run([g_loss, output, summary_op], feed_dict={
        input_images: images,
        target_images: targets,
        })

    test_writer.add_summary(summary, batch_idx)

    for idx, (i,j) in enumerate(batch_range):
        image = Image.fromarray((demo[idx,:,:,0] * 255).astype(np.uint8))
        image.save(os.path.join(output_path, str(i) + '.png'))

    logging.info('Validation Loss: {:.8f}'.format(loss / len(batch_range)))


def g_train_step(batch_idx):
    batch_range = random.sample(train_ids, args.batch_size)

    images, targets = load_batch(batch_range)

    loss, summary, _ = sess.run([g_loss, summary_op, g_optimizer], feed_dict={
        input_images: np.array(images),
        target_images: np.array(targets),
        })

    if batch_idx % train_data_update_freq == 0:
        logging.info('Gen Train: [{}/{} ({:.0f}%)]\tGen Loss: {:.8f}'.format(
            batch_idx, n_iter,
            100. * batch_idx / n_iter, loss))

        train_writer.add_summary(summary, batch_idx)


def d_train_step(batch_idx):
    batch_range = random.sample(train_ids, args.batch_size)

    images, targets = load_batch(batch_range)

    loss, summary, _ = sess.run([d_loss, summary_op, d_optimizer], feed_dict={
        input_images: np.array(images),
        target_images: np.array(targets),
        })

    if batch_idx % train_data_update_freq == 0:
        logging.info('Disc Train: [{}/{} ({:.0f}%)]Disc Loss: {:.8f}'.format(
            batch_idx, n_iter,
            100. * batch_idx / n_iter, loss))

        train_writer.add_summary(summary, batch_idx)


def a_train_step(batch_idx):
    batch_range = random.sample(train_ids, args.batch_size)

    images, targets = load_batch(batch_range)

    loss, summary, _ = sess.run([a_loss, summary_op, a_optimizer], feed_dict={
        input_images: np.array(images),
        target_images: np.array(targets)})

    if batch_idx % train_data_update_freq == 0:
        logging.info('Adv Train: [{}/{} ({:.0f}%)]\tGen Loss: {:.8f}'.format(
            batch_idx, n_iter,
            100. * (batch_idx+1) / n_iter, loss))

        train_writer.add_summary(summary, batch_idx)

batch_idx = 0
while batch_idx < n_iter:
    batch_idx = global_step.eval(sess)

    if batch_idx < 8000:
        g_train_step(batch_idx)
    elif batch_idx < 10000:
        d_train_step(batch_idx)
    else:
        a_train_step(batch_idx)

    if batch_idx % test_data_update_freq == 0:
        test_step(batch_idx)

    if batch_idx % sess_save_freq == 0:
        logging.debug('Saving model')
        saver.save(sess, os.path.join(args.logdir, model_name+".ckpt"), global_step=batch_idx)

