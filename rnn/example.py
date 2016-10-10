#!/usr/bin/env python
# -*- coding: utf_8 -*-

import codecs, copy, sys, time
import cPickle as pickle
from chainer import Variable, FunctionSet, optimizers
import chainer.links as L
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


def train(train_data, words, vocab, n_units=128, learning_rate_decay=0.97, seq_length=1, batch_size=1,
          epochs=10, grad_clip=5, learning_rate_decay_after=2):

    # モデルの構築、初期化
    model = gru.GRU(len(vocab), n_units)
    model = L.Classifier(model)
    model.compute_accuracy = False
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.08, 0.08, data.shape)
    # optimizerの設定
    optimizer = optimizers.RMSprop()
    optimizer.setup(model)

    whole_len = train_data.shape[0]
    jump = whole_len / batch_size
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    state = gru.make_initial_state(n_units, batch_size)
    accum_loss = Variable(np.zeros(()))

    print 'going to train {} iterations'.format(jump * epochs)
    for seq in xrange(jump * epochs):

        input_batch = np.array([train_data[(jump * j + seq) % whole_len]
                            for j in xrange(batch_size)])
        teach_batch = np.array([train_data[(jump * j + seq + 1) % whole_len]
                            for j in xrange(batch_size)])
        input = Variable(input_batch.astype(np.int32), volatile=False)
        teach = Variable(teach_batch.astype(np.int32), volatile=False)

        # 誤差計算
        loss_seq = model(input, teach)
        accum_loss += loss_seq.data

        # 最適化の実行
        if (seq + 1) % seq_length == 0:
            now = time.time()
            print '{}/{}, train_loss = {}, time = {:.2f}'.format((seq + 1) / seq_length, jump,
                                                                 accum_loss.data / seq_length, now - cur_at)
            open('loss', 'w').write('{}\n'.format(accum_loss.data / seq_length))
            cur_at = now

            model.zerograds()
            accum_loss.backward()
            accum_loss.unchain_backward()
            accum_loss = Variable(np.zeros(()))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        # if (seq + 1) % 1000 == 0:
        #    pickle.dump(copy.deepcopy(model).to_cpu(), open('charmodel', 'wb'))

        if (seq + 1) % jump == 0:
            epoch += 1
            if epoch >= learning_rate_decay_after:
                optimizer.lr *= learning_rate_decay
                print 'decayed learning rate by a factor {} to {}'.format(learning_rate_decay, optimizer.lr)

        sys.stdout.flush()

    pickle.dump(copy.deepcopy(model).to_cpu(), open('finalmodel', 'wb'))


if __name__ == "__main__":
    # 学習データの読み込み
    train_date, words, vocab = load_date()
    # vocabの保存
    pickle.dump(vocab, open('vocab.bin', 'wb'))

    # 学習開始
    train(train_date, words, vocab)
