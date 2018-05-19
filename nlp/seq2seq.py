#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable
import numpy as np

"""
https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

"""

UNK = 0
EOS = 1


class Seq2Seq(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            # Embedding Layer
            self.x_embed = L.EmbedID(vocab_size, embed_size)
            self.y_embed = L.EmbedID(vocab_size, embed_size)
            # 2-Layer LSTM
            self.encoder = L.NStepLSTM(n_layers=2, in_size=embed_size, out_size=hidden_size, dropout=0.3)
            self.decoder = L.NStepLSTM(n_layers=2, in_size=embed_size, out_size=hidden_size, dropout=0.3)

            # Attention Layer
            self.attention = L.Linear(2*hidden_size, hidden_size)

            # Output Layer
            self.y = L.Linear(hidden_size, vocab_size)

    def global_attention_layer(self, dec_h, attention):
        """
        https://nlp.stanford.edu/pubs/emnlp15_attn.pdf
        :param dec_h: デコーダの中間層(内部状態)
        :param attention: エンコーダの中間層(内部状態)
        :return:
        """
        weights = F.softmax(F.matmul(dec_h, attention, transb=True))  # Global align weights
        contexts = F.matmul(weights, attention)  # Context vector(Attention layer output)
        o = F.tanh(self.attention(F.concat((contexts, dec_h))))  # Attentionとデコーダの中間層の合成

        return self.y(o)

    def __call__(self, x, y):
        """

        :param x: ミニバッチの入力データ
        :param y: 入力データに対応するミニバッチの出力
        :return: 誤差
        """

        batch_size = len(x)
        eos = np.array([EOS], dtype='int32')

        # EOS信号の埋め込み
        y_in = [F.concat((eos, tmp), axis=0) for tmp in y]
        y_out = [F.concat((tmp, eos), axis=0) for tmp in y]

        # Embedding Layer
        emb_x = [self.x_embed(tmp) for tmp in x]
        emb_y = [self.y_embed(tmp) for tmp in y_in]

        # Encoder, Decoderへの入力
        h, c, a = self.encoder(None, None, emb_x)  # h => hidden, c => cell, a => output(Attention)
        _, _, dec_hs = self.decoder(h, c, emb_y)  # dec_hs=> output

        # Output Layerの計算
        loss = 0
        for dec_h, t, attention in zip(dec_hs, y_out, a):
            # o = self.y(output)
            o = self.global_attention_layer(dec_h, attention)  # Attention Layerの計算
            loss += F.softmax_cross_entropy(o, t)  # 誤差計算
        loss /= batch_size

        return loss

    def predict(self, x, max_length=50):
        """
        テスト用の予測メソッド
        :param x: 入力文
        :param max_length: 出力文の制御
        :return: 出力分のindex
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            result = []
            y = np.array([EOS], dtype='int32')

            # Embedding Layer
            x = x[::-1]  # 入力を反転したほうが精度が上がる
            emb_x = [self.x_embed(x)]

            # Encoder
            h, c, a = self.encoder(None, None, emb_x)

            # Decoder, Output Layerの計算
            for i in range(max_length):
                emb_y = [self.y_embed(y)]  # 初めはEOS信号を次からはDecoderの出力をEmbedding
                h, c, output = self.decoder(h, c, emb_y)  # h => hidden, c => cell, a => output
                o = self.global_attention_layer(output[0], a[0])  # Attention Layerの計算

                # Softmax関数による各単語の生成確率を求める
                prob = F.softmax(o)
                y = np.argmax(prob.data)  # argmaxによってindexを得る

                # 出力がEOS信号かcheck。もしそうならbreak
                if y == EOS:
                    break
                result.append(y)
                # 次の入力へのVariable化
                y = Variable(np.array([y], dtype=np.int32))

            return result
