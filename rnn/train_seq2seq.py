#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function

import cPickle as pickle
import copy

import chainer
from chainer import optimizers

from seq2seq import Seq2Seq
from utils import make_vocab_dict, load_file


def train(input_sentence, output_sentence, n_feat=128, n_hidden=128, iteration=150):
    # vocabularyの作成
    input_vocab = make_vocab_dict(input_sentence)
    output_vocab = make_vocab_dict(output_sentence)
    input_sentence = input_sentence.split()
    output_sentence = output_sentence.split()
    vocab = {}
    # 出力用
    for c, i in output_vocab.items():
        vocab[i] = c

    # modelの構築
    model = Seq2Seq(n_input=len(input_vocab), n_feat=n_feat, n_hidden=n_hidden, n_output=len(output_vocab))
    model.compute_accuracy = False
    model.reset_state()

    # optimizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))  # 勾配の上限

    for i in xrange(iteration):
        for x, y in zip(input_sentence, output_sentence):
            inputs = [input_vocab[word] for word in reversed(x)]
            outputs = [output_vocab[word] for word in y]

            print('入力-> ' + ''.join(x[1:-2]))
            loss = model(inputs, outputs)
            # print('{}/{}, train_loss = {}'.format(i, iteration, loss.data))
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

            for index in model.predict(inputs, outputs):
                print(vocab[index], end='')
            print()

    pickle.dump(copy.deepcopy(model).to_cpu(), open('seq2seq', 'wb'))


if __name__ == "__main__":
    # input_vocab = [u"メリー！ボブスレーしよう！！"]
    # output_vocab = [u"オッケー蓮子！！"]

    # テキストの読み込み
    inputs = load_file('train_data\\player1.txt').replace('\r', u'eos')
    outputs = load_file('train_data\\player2.txt').replace('\r', u'eos')

    train(input_sentence=inputs, output_sentence=outputs)
