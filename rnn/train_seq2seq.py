#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function

import cPickle as pickle
import copy
import time

import chainer
from chainer import optimizers
from mecab import mecab_wakati
from seq2seq import Seq2Seq
from utils import make_vocab_dict, load_file, plt_loss
import matplotlib.pyplot as plt


def train(input_sentence, output_sentence, n_feat=128, n_hidden=128, iteration=50):
    # vocabularyの作成
    in_set, input_vocab = make_vocab_dict(input_sentence)
    out_set, output_vocab = make_vocab_dict(output_sentence)
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

    # 表示用
    cur_at = time.time()
    loss_plot = []
    epoch = 1

    # train
    for i in range(iteration):
        for x, y in zip(input_sentence, output_sentence):
            inputs = [input_vocab[word] for word in reversed(mecab_wakati(x).split())]
            outputs = [output_vocab[word] for word in mecab_wakati(y).split()]

            # print('入力-> ' + ''.join(x[0:-2]))
            loss = model(inputs, outputs)
            loss_plot.append(loss.data)
            now = time.time()
            print('{}/{}, train_loss = {}, time = {:.2f}'.format(
                epoch, len(input_sentence)*iteration, loss.data, now-cur_at))
            epoch += 1
            cur_at = now
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

            # for index in model.predict(inputs, output_vocab):
            #    print(vocab[index], end='')
            # print()
    # テスト用
    for sentence in input_sentence:
        # test = "メリー！ボブスレーしよう！！"
        inputs = [input_vocab[word] for word in reversed(mecab_wakati(sentence).split())]
        print(u'入力 -> ' + sentence)
        print(u'出力 -> ', end='')
        for index in model.predict(inputs, output_vocab):
            print(vocab[index], end='')
        print()

    plt_loss(loss_plot)
    pickle.dump(copy.deepcopy(model).to_cpu(), open('seq2seq', 'wb'))


if __name__ == "__main__":
    # input_vocab = [u"メリー！ボブスレーしよう！！"]
    # output_vocab = [u"オッケー蓮子！！"]

    # テキストの読み込み
    inputs = load_file('train_data\\player1.txt')
    outputs = load_file('train_data\\player2.txt').replace('\r', u'eos')

    train(input_sentence=inputs, output_sentence=outputs)
