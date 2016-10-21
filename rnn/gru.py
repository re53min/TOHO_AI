#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as linear
from chainer import link
import numpy as np


class GRU(link.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(GRU, self).__init__(
            embed=linear.EmbedID(n_vocab, n_units),
            l1=linear.StatefulGRU(n_units, n_units),
            l2=linear.StatefulGRU(n_units, n_units),
            l3=linear.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.08, 0.08, param.data.shape)
        self.train = train

    def __call__(self, x):

        h0 = self.embed(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)

        return y

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

