#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function
import cPickle as pickle
import codecs
import sys

import chainer
from chainer import cuda
import chainer.functions as function
import numpy as np
import six

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


def load(model, vocab):
    # Vocablaryの読み込み
    vocab = pickle.load(open(vocab, "rb"))
    ivocab = {}

    for c, i in vocab.items():
        ivocab[i] = c
    # Modelの読み込み
    model = pickle.load(open(model, "rb"))

    return vocab, ivocab, model


def predict(model="model", vocab="vocab.bin", length=5000, sample=0):
    # ロード
    vocab, ivocab, model = load(model, vocab)
    xp = np
    model.predictor.reset_state()

    # 標準入力
    # print 'Please your typing!!'
    input_text = u"秋風"
    if isinstance(input_text, six.binary_type):
        input_text = input_text.decode('utf-8')
    # 入力された文字がvocabの中に含まれていたらprev_wordの生成
    if input_text in vocab:
        prev_word = chainer.Variable(xp.array([vocab[input_text]], xp.int32))
    else:
        print('Error: Unfortunately ' + input_text + ' is unknown')
        exit()
    # 初めの一文字の出力
    sys.stdout.write(input_text + ' ')
    prob = function.softmax(model.predictor(prev_word))

    for i in xrange(length):
        # 次の単語の予測
        prob = function.softmax(model.predictor(prev_word))

        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        # eosタグが予測された場合に読点に置換
        if ivocab[index] == u'eos':
            sys.stdout.write('\r\n')
        else:
            sys.stdout.write(ivocab[index] + ' ')
        # 次の文字へ
        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')


if __name__ == "__main__":
    predict()
