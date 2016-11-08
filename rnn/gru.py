#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as linear
import chainer.functions as function
from chainer import link
import numpy as np


class GRU(link.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(GRU, self).__init__(
            embed=linear.EmbedID(n_vocab, n_units, ignore_label=-1),
            l1=linear.StatefulGRU(n_units, n_units),
            l2=linear.StatefulGRU(n_units, n_units),
            l3=linear.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def __call__(self, x):

        h0 = self.embed(x)
        h1 = self.l1(function.dropout(h0, train=self.train))
        h2 = self.l2(function.dropout(h1, train=self.train))
        y = self.l3(function.dropout(h2, train=self.train))

        return y

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

