#!/usr/bin/env python
# -*- coding: utf_8 -*-

import io
import pickle
import copy
import sys
import time

import chainer
import chainer.links as L
import numpy as np
from chainer import Variable, optimizers

from NLP.gru import GRU
from utils.utils import make_vocab_dict, load_file
from utils.visualizer import plot_loss

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def train(train_data, vocab, n_units=300, learning_rate_decay=0.97, seq_length=20, batch_size=20,
          epochs=20, learning_rate_decay_after=5):
    # モデルの構築、初期化
    model = L.Classifier(GRU(len(vocab), n_units))
    model.compute_accuracy = False

    # optimizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))  # 勾配の上限

    whole_len = train_data.shape[0]
    jump = whole_len / batch_size
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    loss = 0
    plt_loss = []

    print('going to train {} iterations'.format(jump * epochs))
    for seq in range(jump * epochs):

        input_batch = np.array([train_data[(jump * j + seq) % whole_len]
                                for j in range(batch_size)])
        teach_batch = np.array([train_data[(jump * j + seq + 1) % whole_len]
                                for j in range(batch_size)])
        x = Variable(input_batch.astype(np.int32), volatile=False)
        teach = Variable(teach_batch.astype(np.int32), volatile=False)

        # 誤差計算
        loss += model(x, teach)

        # 最適化の実行
        if (seq + 1) % seq_length == 0:
            now = time.time()
            plt_loss.append(loss.data)
            print('{}/{}, train_loss = {}, time = {:.2f}'.format((seq + 1) / seq_length, jump,
                                                                 loss.data / seq_length, now - cur_at))
            # open('loss', 'w').write('{}\n'.format(loss.data / seq_length))
            cur_at = now

            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            loss = 0

        # check point
        if (seq + 1) % 10000 == 0:
            pickle.dump(copy.deepcopy(model).to_cpu(), open('check_point', 'wb'))

        if (seq + 1) % jump == 0:
            epoch += 1
            if epoch >= learning_rate_decay_after:
                # optimizer.lr *= learning_rate_decay
                print('decayed learning rate by a factor {} to {}'.format(learning_rate_decay, optimizer.lr))

        sys.stdout.flush()

    pickle.dump(copy.deepcopy(model).to_cpu(), open('rnnlm_model', 'wb'))
    plot_loss(plt_loss)


if __name__ == "__main__":
    # 学習データの読み込み
    file_path = 'train_data\\train_renko.txt'
    train_date, vocab = make_vocab_dict(load_file(file_path))
    # vocabの保存
    pickle.dump(vocab, open('rnnlm_vocab.bin', 'wb'))
    # 学習開始
    train(train_date, vocab)
