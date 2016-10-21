#!/usr/bin/env python
# -*- coding: utf_8 -*-

import cPickle as pickle
import codecs
import sys

import chainer
import chainer.links as linear
from chainer import serializers
from chainer import cuda
import chainer.functions as function
import numpy as np
import six

import gru

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


def predict(model="model", vocab="vocab.bin", length=2000, sample=1):
    # ロード
    vocab, ivocab, model = load(model, vocab)
    xp = np
    # n_units = model.embed.W.data.shape[1]
    # lm = gru.GRU(len(vocab), n_units, train=False)
    # lm = linear.Classifier(lm)
    # serializers.load_npz(model, lm)

    model.predictor.reset_state()

    # 標準入力
    print 'Please your typing!!'
    input_text = "あ"
    if isinstance(input_text, six.binary_type):
        input_text = input_text.decode('utf-8')
    # 入力された文字がvocabの中に含まれていたらprev_wordの生成
    if input_text in vocab:
        prev_word = chainer.Variable(np.array([vocab[input_text]], np.int32))
    else:
        print 'Error: Unfortunately ' + input_text + ' is unknown'
        exit()

    # prob = function.softmax(model.predictor(prev_word))
    sys.stdout.write(input_text + ' ')

    for i in six.moves.range(length):
        prob = function.softmax(model.predictor(prev_word))

        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        if ivocab[index] == '<eos>':
            sys.stdout.write('。')
        else:
            sys.stdout.write(ivocab[index] + ' ')

        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')


if __name__ == "__main__":
    predict()
