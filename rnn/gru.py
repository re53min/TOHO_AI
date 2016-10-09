#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import link, Variable
import numpy as np



class GRU(link.Chain):
    def __init__(self, n_vocab, n_units):
        super(GRU, self).__init__(
            embed=L.EmbedID(n_vocab, 4 * n_units),
            l1=L.StatefulGRU(n_vocab, 4 * n_units),
            l2=L.StatefulGRU(n_vocab, 4 * n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        #for param in self.parameter:
        #    param[:] = np.random.uniform(-0.08, 0.08, param.shape)

    def __call__(self, x):

        h0 = self.embed(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)

        return y

    def reset_state(self):
        self.l1.reset_state()


def make_initial_state(n_units, batchsize=50, train=True):
    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32),
                           volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}
