# -*- coding: utf_8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable
import numpy as np

"""
https://www.aclweb.org/anthology/N13-1090
"""

UNK = 0
EOS = 1


class RNNLM(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNLM, self).__init__()
        with self.init_scope():
            # Embedding Layer
            self.embed = L.EmbedID(vocab_size, embed_size)

            # LSTM
            self.h = L.LSTM(in_size=embed_size, out_size=hidden_size)
            # 2-Layer LSTM
            # self.h = L.NStepLSTM(n_layers=2, in_size=embed_size, out_size=hidden_size, dropout=0.3)

            # Output Layer
            self.y = L.Linear(hidden_size, vocab_size)

    def __call__(self, x):

        batch_size = len(x)
        eos = np.array([EOS], dtype='int32')

        # EOS信号の埋め込み
        in_x = [F.concat((eos, tmp), axis=0) for tmp in x]
        in_y = [F.concat((tmp, eos), axis=0) for tmp in x]

        # Embedding Layer
        emb_x = [self.embed(tmp) for tmp in in_x]

        # LSTMへの入力
        _, _, outputs = self.h(None, None, emb_x)  # h => hidden, c => cell, a => output(Attention)

        # Output Layerの計算
        loss = 0
        for output, t in zip(outputs, in_y):
            o = self.y(output)
            # print(o.shape[0])
            # print(t[1:].shape[0])
            loss += F.softmax_cross_entropy(o, t)  # 誤差計算
        loss /= batch_size

        return loss

    def predict(self, x, max_length=1000):
        """
        テスト用の予測メソッド
        :param x: 入力文
        :param max_length: 出力文の制御
        :return: 出力分のindex
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            result = []
            h = None
            c = None

            # Decoder, Output Layerの計算
            for i in range(max_length):
                # Embedding Layer
                x = [self.embed(x)]

                # Hidden Layer
                h, c, output = self.h(h, c, x)  # h => hidden, c => cell, a => output
                o = self.y(output[0])

                # Softmax関数による各単語の生成確率を求める
                prob = F.softmax(o)
                y = np.argmax(prob.data)  # argmaxによってindexを得る

                # 出力がEOS信号かcheck。もしそうならbreak
                # if y == EOS:
                #    break
                result.append(y)
                # 次の入力へのVariable化
                x = Variable(np.array([y], dtype=np.int32))

            return result
