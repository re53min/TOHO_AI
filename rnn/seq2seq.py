#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import link
from chainer import Variable, optimizers
from chainer import cuda


def make_vocab_dict(words):
    vocab = {}  # Word ID
    # dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
            # print(str(word))
            # デーアセットにwordを登録
            # dataset[i] = vocab[word]

    print("corpus size: ", len(words))
    print("vocabulary size: ", len(vocab))

    return vocab  # dataset, vocab


class Seq2Seq(link.Chain):
    def __init__(self, n_input, n_feat, n_hidden, n_output, train=True):
        super(Seq2Seq, self).__init__(
            # encoder layer
            input_embed=L.EmbedID(n_input, n_feat, ignore_label=-1),
            encode1=L.StatefulGRU(n_hidden, n_feat),
            encode2=L.StatefulGRU(n_hidden, n_hidden),
            # decoder layer
            output_embed=L.EmbedID(n_output, n_feat, ignore_label=-1),
            decode1=L.StatefulGRU(n_hidden, n_feat),
            decode2=L.StatefulGRU(n_hidden, n_hidden),
            output=L.Linear(n_hidden, n_output),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.08, 0.08, param.data.shape)
        self.train = train

    def reset_state(self):
        self.encode1.reset_state()
        self.encode2.reset_state()
        self.decode1.reset_state()
        self.decode2.reset_state()

    def encode(self, sentences):

        for word in sentences:
            word = Variable(np.array([[word]], dtype=np.int32))
            input_embed = self.input_embed(word)
            enc1 = self.encode1(input_embed)
            enc2 = self.encode2(enc1)

        return enc2

    def decode(self, sentences):
        # sentences = Variable(np.array([sentences], dtype=np.int32), volatile=False)
        loss = Variable(np.zeros((), dtype=np.float32))
        n_words = len(sentences)-1

        for word, t in zip(sentences, sentences[1:]):
            # print('入力:{}, 教師:{}'.format(word,t))
            word = Variable(np.array([[word]], dtype=np.int32))
            t = Variable(np.array([t], dtype=np.int32))
            decode0 = self.output_embed(word)
            decode1 = self.decode1(decode0)
            decode2 = self.decode2(decode1)
            z = self.output(decode2)

            loss += F.softmax_cross_entropy(z, t)

        return loss, n_words

    def init_decoder(self, h_enc1, h_enc2):
        self.decode1.set_state(h_enc1)
        self.decode2.set_state(h_enc2)

    def test_decode(self, start, eos, limit):
        output = []
        y = chainer.Variable(np.array([[start]], dtype=np.int32))

        for i in xrange(limit):
            decode0 = self.output_embed(y)
            decode1 = self.decode1(decode0)
            decode2 = self.decode2(decode1)
            z = self.output(decode2)
            prob = F.softmax(z)

            index = np.argmax(cuda.to_cpu(prob.data))

            if index == eos:
                break
            output.append(index)
            y = chainer.Variable(np.array([index], dtype=np.int32))
        return output

    def predict(self, input_sentence, output_sentence):
        limit = 5
        bos_id = output_sentence[0]
        eos_id = output_sentence[-1]

        self.reset_state()
        self.encode(input_sentence)
        self.init_decoder(
            self.encode1.h,
            self.encode2.h
        )
        z = self.test_decode(bos_id, eos_id, limit)

        return z

    def __call__(self, x, t):
        # encode
        self.encode(x)
        # decode
        self.init_decoder(
            self.encode1.h, self.encode2.h
        )
        loss, n_word = self.decode(t)

        return loss

if __name__ == "__main__":

    # input_vocab = [u"メリー！ボブスレーしよう！！"]
    # output_vocab = [u"オッケー蓮子！！"]

    input_sentence = ["メリー", "！", "ボブスレー", "しよ", "う", "！", "！"]
    output_sentence = ["<start>", "オッケー", "蓮子", "！", "！"] + ["<eos>"]

    input_vocab = make_vocab_dict(input_sentence)  # inputs, input_vocab = make_vocab_dict(input_sentence)
    output_vocab = make_vocab_dict(output_sentence)  # outputs, output_vocab = make_vocab_dict(output_sentence)
    vocab = {}

    model = Seq2Seq(n_input=len(input_vocab), n_feat=10, n_hidden=10, n_output=len(output_vocab))
    model.compute_accuracy = False

    # optimizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))  # 勾配の上限

    loss = 0
    iteration = 150
    model.reset_state()
    inputs = [input_vocab[word] for word in reversed(input_sentence)]
    outputs = [output_vocab[word] for word in output_sentence]

    for c, i in output_vocab.items():
        vocab[i] = c

    print('入力-> ' + ''.join(input_sentence[1:-2]))

    for i in xrange(iteration):

        loss = model(inputs, outputs)
        # print('{}/{}, train_loss = {}'.format(i, iteration, loss.data))
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        for index in model.predict(inputs, outputs):
            print(vocab[index], end='')
        print()
