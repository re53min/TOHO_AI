#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import link
from chainer import Variable, optimizers


def make_vocab_dict(words):
    vocab = {}  # Word ID
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
            # print(str(word))
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print("corpus size: ", len(words))
    print("vocabulary size: ", len(vocab))

    return dataset, vocab


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
            output_embed=L.EmbedID(h_decode, n_feat),
            output=L.Linear(n_feat, n_output),
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

    def decode(self, context):

        decode0 = self.decode1(context)
        decode1 = self.decode2(decode0)
        ouput_embded = F.tanh(self.output_embed(decode1))
        output = self.output(ouput_embded)

        return output


if __name__ == "__main__":
    # input_vocab = [u"メリー！ボブスレーしよう！！"]
    # output_vocab = [u"オッケー蓮子！！"]

    input_sentence = [u"メリー", u"！", u"ボブスレー", u"しよ", u"う", u"！", u"！"]
    output_sentence = [u"オッケー", u"蓮子", u"！", u"！"]

    input, input_vocab = make_vocab_dict(input_sentence)
    output, output_vocab = make_vocab_dict(output_sentence)

    model = L.classifier(
        Seq2Seq(n_input=len(input_vocab), n_feat=4, h_encode=10, h_decode=10, n_output=len(output_vocab)))
    model.compute_accuracy = False

    # optimizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))  # 勾配の上限

    # for i in xrange(20000):
