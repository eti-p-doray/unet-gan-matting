#!/usr/bin/python3

import argparse
import os
import random
import logging

import tensorflow as tf
import numpy as np
from PIL import Image

from unet import UNet, Discriminator
from scripts.image_manips import resize

model_name = "matting"

logging.basicConfig(level=logging.INFO)

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Trains the unet")
    parser.add_argument("data", type=str,
        help="Path to a folder containing data to train")
    parser.add_argument("--lr", type=float, default=1.0,
        help="Learning rate used to optimize")
    parser.add_argument("--d_coeff", type=float, default=1.0,
        help="Discriminator loss coefficient")
    parser.add_argument("--gen_epoch", type=int, default=4,
        help="Number of training epochs")
    parser.add_argument("--disc_epoch", type=int, default=1,
        help="Number of training epochs")
    parser.add_argument("--adv_epoch", type=int, default=5,
        help="Number of training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4,
        help="Size of the batches used in training")
    parser.add_argument('--checkpoint', type=int, default=None,
        help='Saved session checkpoint, -1 for latest.')
    parser.add_argument('--logdir', default="log/" + model_name,
        help='Directory where logs should be written.')
    return  parser.parse_args()


def apply_trimap(images, output, alpha):
    masked_output = []
    for channel in range(4):
        masked_output.append(output[:,:,:,channel])
        masked_output[channel] = tf.where(alpha < 0.25, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = tf.where(alpha > 0.75, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = masked_output[channel]
    masked_output = tf.stack(masked_output, 3)
    return masked_output

def main(args):
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

    g_iter = int(args.gen_epoch * int(train_ids.shape[0]))
    d_iter = int(args.disc_epoch * int(train_ids.shape[0]))
    a_iter = int(args.adv_epoch * int(train_ids.shape[0]))
    n_iter = g_iter+d_iter+a_iter


    input_images = tf.placeholder(tf.float32, shape=[None, 480, 360, 4])
    target_images = tf.placeholder(tf.float32, shape=[None, 480, 360, 4])
    alpha = target_images[:,:,:,3][..., np.newaxis]

    with tf.variable_scope("Gen"):
        gen = UNet(4,4)
        output = tf.sigmoid(gen(input_images))
        g_loss = tf.losses.mean_squared_error(target_images, output)
    with tf.variable_scope("Disc"):
        disc = Discriminator(4)
        d_real = disc(target_images)
        d_fake = disc(output)
        d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))

    a_loss = g_loss + args.d_coeff * d_loss

    g_loss_summary = tf.summary.scalar("g_loss", g_loss)
    d_loss_summary = tf.summary.scalar("d_loss", d_loss)
    a_loss_summary = tf.summary.scalar("a_loss", a_loss)

    summary_op = tf.summary.merge(
        [g_loss_summary, d_loss_summary, a_loss_summary])

    summary_image = tf.summary.image("result", output)

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

    def load_batch(batch_ids):
        images, targets = [], []
        for i, j in batch_ids:
            input_filename = os.path.join(input_path, str(i) + '_' + str(j) + '.jpg')
            trimap_filename = os.path.join(trimap_path, str(i) + '_trimap.jpg')
            target_filename = os.path.join(target_path, str(i) + '.png')
            logging.debug(input_filename)
            logging.debug(trimap_filename)
            logging.debug(target_filename)
            image = resize(Image.open(input_filename), 2)
            trimap = resize(Image.open(trimap_filename), 2)
            target = resize(Image.open(target_filename), 2)

            image = np.array(image)
            trimap = np.array(trimap)[..., np.newaxis]
            image = np.concatenate((image, trimap), axis = 2).astype(np.float32) / 255

            target = np.array(target).astype(np.float32) / 255

            images.append(image)
            targets.append(target)

        return np.asarray(images), np.asarray(targets)


    def test_step(batch_idx, summary_fct):
        batch_range = random.sample(train_ids, args.batch_size)

        images, targets = load_batch(batch_range)

        loss, demo, summary = sess.run([g_loss, summary_image, summary_fct], feed_dict={
            input_images: images,
            target_images: targets,
            })

        test_writer.add_summary(summary, batch_idx)
        test_writer.add_summary(demo, batch_idx)

        logging.info('Validation Loss: {:.8f}'.format(loss))

    try:
        batch_idx = 0
        while batch_idx < n_iter:
            batch_idx = global_step.eval(sess) * args.batch_size

            loss_fct = None
            label = None
            optimizers = []
            if batch_idx < g_iter:
                loss_fct = g_loss
                summary_fct = g_loss_summary
                label = 'Gen train'
                optimizers = [g_optimizer]
            elif batch_idx < g_iter+d_iter:
                loss_fct = d_loss
                summary_fct = d_loss_summary
                label = 'Disc train'
                optimizers = [d_optimizer]
            else:
                loss_fct = a_loss
                summary_fct = summary_op
                label = 'Adv train'
                optimizers = [a_optimizer]

            batch_range = random.sample(train_ids, args.batch_size)
            images, targets = load_batch(batch_range)

            loss, summary = sess.run([loss_fct, summary_fct] +  optimizers, feed_dict={
                input_images: np.array(images),
                target_images: np.array(targets)})[0:2]

            if batch_idx % train_data_update_freq == 0:
                logging.info('{}: [{}/{} ({:.0f}%)]\tGen Loss: {:.8f}'.format(label, batch_idx, n_iter,
                    100. * (batch_idx+1) / n_iter, loss))

                train_writer.add_summary(summary, batch_idx)

            if batch_idx % test_data_update_freq == 0:
                test_step(batch_idx, summary_fct)

            if batch_idx % sess_save_freq == 0:
                logging.debug('Saving model')
                saver.save(sess, os.path.join(args.logdir, model_name+".ckpt"), global_step=batch_idx)

    except Exception:
        saver.save(sess, os.path.join(args.logdir, 'crash_save_'+model_name+".ckpt"), global_step=batch_idx)


    saver.save(sess, os.path.join(args.logdir, model_name+".ckpt"), global_step=batch_idx)


if __name__ == '__main__':
    args = parse_args()
    main(args)
