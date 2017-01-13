#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import link
import numpy as np


class ImageNet(link.Chain):
    def __init__(self, n_outputs, train=True):
        super(ImageNet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=4),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(None, 128, 5, pad=2),
            bn2=L.BatchNormalization(128),
            conv3=L.Convolution2D(None, 256, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),

            l5=L.Linear(None, 512),
            l6=L.Linear(512, n_outputs),

        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

