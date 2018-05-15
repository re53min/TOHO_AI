#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable
import numpy as np

"""
https://arxiv.org/pdf/1609.05473.pdf

"""

UNK = 0
EOS = 1


class Generator(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Generator, self).__init__()
        with self.init_scope():
            self.x_embed = L.EmbedID(vocab_size, embed_size)
            self.encoder = L.NStepLSTM(n_layers=1, in_size=embed_size, out_size=hidden_size, dropout=0.3)
            self.y = L.Linear(hidden_size, vocab_size)

    def __call__(self, x):

        # batch_size = len(x)
        eos = np.array([EOS], dtype='int32')
        x = [F.concat((tmp, eos), axis=0) for tmp in x]

        # Embedding Layer
        emb_x = [self.x_embed(tmp) for tmp in x]

        # Encoder, Decoderへの入力
        _, _, outputs = self.encoder(None, None, emb_x)  # h => hidden, c => cell, a => output(Attention)

        # Output Layerの計算
        result = []
        for output in outputs:
            o = self.y(output)
            prob = F.softmax(o)
            y = np.argmax(prob.data)

            # 出力がEOS信号かcheck。もしそうならbreak
            if y == EOS:
                break
            result.append(y)

        return result


class Discriminator(Chain):

    def __init__(self, num_classes, vocab_size, embed_size, filter_size, num_filter):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.x_embed = L.EmbedID(vocab_size, embed_size)
            self.conv1 = L.Convolution2D(1, num_filter, (filter_size, embed_size))
            self.pooling = L.Linear(num_filter, num_filter)
            self.y = L.Linear(num_filter, num_classes)

    def __call__(self, x):
        # Embedding Layer
        emb_x = [self.x_embed(tmp) for tmp in x]

        #
        h1 = F.max_pooling_2d(F.relu(self.conv1(emb_x)), stride=1)
        h2 = F.dropout(F.relu(h1))
        o = self.y(h2)

        y = F.softmax(o)

        return y