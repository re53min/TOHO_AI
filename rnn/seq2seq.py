#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import link


class Seq2Seq(link.Chain):
    def __init__(self, n_input, n_output, n_feat, h_encode, h_decode, train=True):
        super(Seq2Seq, self).__init__(
            # encoder layer
            input=L.EmbedID(n_input, n_feat, ignore_label=-1),
            encode1=L.StatefulGRU(n_feat, h_encode),
            encode2=L.StatefulGRU(h_encode, h_encode),
            # connection layer
            connection=L.StatefulGRU(h_encode, h_decode),
            # decoder layer
            decode1=L.StatefulGRU(h_decode, h_decode),
            decode2=L.StatefulGRU(h_decode, h_decode),
            output=L.Linear(h_decode, n_output),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

