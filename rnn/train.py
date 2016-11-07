#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function
import cPickle as pickle
import codecs
import copy
import sys
import time

import chainer.links as linear
import numpy as np
from chainer import Variable, optimizers

import gru
from mecab import mecab_wakati


def load_date(mecab=True):
    vocab = {}  # Word ID
    with codecs.open('train_data.txt', 'r', 'utf-8') as sentences:  # .replace('\r\n', '<eos>')  # textの読み込み
        sentences = sentences.read()
        # 分かち書き処理
        if mecab:
            words = mecab_wakati(sentence=sentences).replace(u'\r', u'\r\n').split(" ")
        else:
            words = list(sentences)
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print("corpus size: ", len(words))
    print("vocabulary size: ", len(vocab))

    return dataset, vocab


def train(train_data, vocab, n_units=50, learning_rate_decay=0.97, seq_length=20, batch_size=20,
          epochs=50, learning_rate_decay_after=5):
    # モデルの構築、初期化
    model = linear.Classifier(gru.GRU(len(vocab), n_units))
    model.compute_accuracy = False

    # optimizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # optimizer.add_hook(optimizer.GradientCliping(grad_clip))

    whole_len = train_data.shape[0]
    jump = whole_len / batch_size
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    accum_loss = 0

    print('going to train {} iterations'.format(jump * epochs))
    for seq in xrange(jump * epochs):

        input_batch = np.array([train_data[(jump * j + seq) % whole_len]
                                for j in xrange(batch_size)])
        teach_batch = np.array([train_data[(jump * j + seq + 1) % whole_len]
                                for j in xrange(batch_size)])
        x = Variable(input_batch.astype(np.int32), volatile=False)
        teach = Variable(teach_batch.astype(np.int32), volatile=False)

        # 誤差計算
        loss_seq = optimizer.target(x, teach)
        accum_loss += loss_seq

        # 最適化の実行
        if (seq + 1) % seq_length == 0:
            now = time.time()
            print('{}/{}, train_loss = {}, time = {:.2f}'.format((seq + 1) / seq_length, jump,
                                                                 accum_loss.data / seq_length, now - cur_at))
            open('loss', 'w').write('{}\n'.format(accum_loss.data / seq_length))
            cur_at = now

            optimizer.target.cleargrads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = 0
            # optimizer.clip_grads(grad_clip)
            optimizer.update()

        # check point
        if (seq + 1) % 10000 == 0:
           pickle.dump(copy.deepcopy(model).to_cpu(), open('charmodel', 'wb'))

        if (seq + 1) % jump == 0:
            epoch += 1
            if epoch >= learning_rate_decay_after:
                # optimizer.lr *= learning_rate_decay
                print('decayed learning rate by a factor {} to {}'.format(learning_rate_decay, optimizer.lr))

        sys.stdout.flush()

    pickle.dump(copy.deepcopy(model).to_cpu(), open('model', 'wb'))


if __name__ == "__main__":
    # 学習データの読み込み
    train_date, vocab = load_date()
    # vocabの保存
    pickle.dump(vocab, open('vocab.bin', 'wb'))
    # 学習開始
    train(train_date, vocab)
