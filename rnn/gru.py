#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import link
import numpy as np


class GRU(link.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(GRU, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1),
            l1=L.StatefulGRU(n_units, n_units),
            l2=L.StatefulGRU(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def __call__(self, x):

        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))

        return y

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

