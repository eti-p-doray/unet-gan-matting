#!/usr/bin/python3
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        n_channels = [
            8,
            16,
            32,
            64,
        ]

        self.inc = inconv(in_ch, n_channels[0])
        self.down1 = down(n_channels[0], n_channels[1])
        self.down2 = down(n_channels[1], n_channels[2])
        self.down3 = down(n_channels[2], n_channels[3])
        self.up4 = up(n_channels[3], n_channels[2])
        self.up5 = up(n_channels[2], n_channels[1])
        self.up6 = up(n_channels[1], n_channels[0])
        self.outc = outconv(n_channels[0], out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up4(x4, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.outc(x)
        return x
