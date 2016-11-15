#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import link


class Seq2Seq(link.Chain):
    def __init__(self, n_input, n_feat, h_encode, h_decode, n_output, train=True):
        super(Seq2Seq, self).__init__(
            # encoder layer
            input_embed=L.EmbedID(n_input, n_feat, ignore_label=-1),
            encode1=L.StatefulGRU(n_feat, h_encode),
            encode2=L.StatefulGRU(h_encode, h_encode),
            # connection layer
            decode1=L.StatefulGRU(h_encode, h_decode),
            # decoder layer
            decode2=L.StatefulGRU(h_decode, h_decode),
            output_embed=L.EmbedID(n_output, n_feat),
            output=L.Linear(h_decode, n_output),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.encode1.reset_state()
        self.encode2.reset_state()
        self.decode1.reset_state()
        self.decode2.reset_state()

    def encode(self, input_sentence):

        for word in input_sentence:
            input_embed = F.tanh(self.input_embed(word))
            enc1 = self.encode1(input_embed)
            enc2 = self.encode2(enc1)

        return enc2

    def decode(self, enc):

        decode0 = self.decode1(enc)
        decode1 = self.decode2(decode0)
        ouput_embded = F.tanh(self.output_embed(decode1))
        output = self.output(ouput_embded)
        
        return output

