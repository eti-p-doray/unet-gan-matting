#!/usr/bin/python3
# full assembly of the sub-parts to form the complete net

import tensorflow as tf
import numpy as np

# python 3 confusing imports :(
from .unet_parts import *


class Conv():
    def __init__(self, in_ch, out_ch):
        self.out_ch = out_ch

    def __call__(self, x):
        x = tf.layers.conv2d(x, filters=self.out_ch, kernel_size=3, padding="same")
        x = tf.contrib.layers.batch_norm(x)
        x = tf.nn.relu(x)
        return x


class DownConv():
    def __init__(self, in_ch, out_ch):
        self.out_ch = out_ch

    def __call__(self, x):
        x = tf.layers.conv2d(x, filters=self.out_ch, kernel_size=3, padding="same")
        x = tf.contrib.layers.batch_norm(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2)
        x = tf.nn.relu(x)
        return x


class UNet():
    def __init__(self, in_ch, out_ch):
        n_channels = [
            16,
            32,
            64,
            128,
        ]

        self.inc = InConv(in_ch, n_channels[0])
        self.down1 = Down(n_channels[0], n_channels[1])
        self.down2 = Down(n_channels[1], n_channels[2])
        self.down3 = Down(n_channels[2], n_channels[3])
        self.conv1 = Conv(n_channels[3], n_channels[3])
        self.conv2 = Conv(n_channels[3], n_channels[3])
        self.conv3 = Conv(n_channels[3], n_channels[3])
        self.up4 = Up(n_channels[3], n_channels[2])
        self.up5 = Up(n_channels[2], n_channels[1])
        self.up6 = Up(n_channels[1], n_channels[0])
        self.outc = OutConv(n_channels[0], out_ch)

    def __call__(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.conv1(x4)
        x4 = self.conv2(x4)
        x4 = self.conv3(x4)
        x = self.up4(x4, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outc(x)
        return x

class Discriminator():
    def __init__(self, in_ch):
        n_channels = [
            16,
            32,
            64,
            128,
        ]

        self.conv1 = DownConv(in_ch, n_channels[0])
        self.conv2 = DownConv(n_channels[0], n_channels[1])
        self.conv3 = DownConv(n_channels[1], n_channels[2])
        self.conv4 = DownConv(n_channels[2], n_channels[3])

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        dim = np.prod(x.shape[1:])
        x = tf.reshape(x, [-1, dim])
        x = tf.layers.dense(x, 1)
        x = tf.sigmoid(x)
        return x

