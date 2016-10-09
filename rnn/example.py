#!/usr/bin/env python
# -*- coding: utf_8 -*-

import codecs, os.path
import cPickle as pickle
from chainer import Variable, FunctionSet, optimizers
import numpy as np
import gru


def load_date():
    vocab = {}  # Word ID
    words = codecs.open('train_data.txt', 'r', 'utf-8').read()  # textの読み込み
    words = list(words)
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print "corpus size: ", len(words)
    print "vocabulary size: ", len(vocab)

    return dataset, words, vocab


def train(train_data, words, vocab, n_units=128, seq_length=50, batch_size=50, epochs=50, grad_clip=5):
    # モデルの構築、初期化
    model = gru.GRU(len(vocab), n_units)
    optimizer = optimizers.RMSprop()
    optimizer.setup(model)

    whole_len = train_data.shape[0]
    jump = whole_len / batch_size
    epoch = 0
    state = gru.make_initial_state(n_units, batch_size)
    accum_loss = Variable(np.zeros(()))

    for seq in xrange(jump * epochs):
        x_batch = np.array([train_data[(jump * j) % whole_len]
                            for j in xrange(batch_size)])
        y_batch = np.array([train_data[(jump * j + 1) % whole_len]
                            for j in xrange(batch_size)])
        x = Variable(x_batch.astype(np.int32), volatile=False)
        t = Variable(y_batch.astype(np.int32), volatile=False)

        loss_i = model(x, t)

if __name__ == "__main__":

    # 学習データの読み込み
    train_date, words, vocab = load_date()
    # vocabの保存
    pickle.dump(vocab, open('vocab.bin', 'wb'))

    # 学習開始
    train(train_date, words, vocab)
