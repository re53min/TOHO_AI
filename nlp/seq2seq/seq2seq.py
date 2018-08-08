#!/usr/bin/env python
# -*- coding: utf_8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain

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
        eos = self.xp.array([EOS], dtype='int32')

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
        accuracy = 0
        for dec_h, t, attention in zip(dec_hs, y_out, a):
            # o = self.y(dec_h)
            o = self.global_attention_layer(dec_h, attention)  # Attention Layerの計算
            loss += F.softmax_cross_entropy(o, t)  # 誤差計算
            accuracy += F.accuracy(o, t)  # 精度計算
        loss /= batch_size
        accuracy /= batch_size

        return loss, accuracy

    def generate_one_step(self, one_hot, h, c, a):

        """
        1ステップごとにDecoderを出力を得るメソッド
        :param one_hot: word index
        :param h: hidden state
        :param c: cell state
        :param a: attention
        :return: word probabilities, hidden state, cell state
        """

        y = self.xp.array(one_hot, dtype='int32')
        emb_y = [self.y_embed(y)]  # 初めはEOS信号を次からはDecoderの出力をEmbedding
        h, c, dec_h = self.decoder(h, c, emb_y)  # h => hidden, c => cell, a => output
        # o = self.y(dec_h[0])
        o = self.global_attention_layer(dec_h[0], a[0])  # Attention Layerの計算
        prob = F.softmax(o)

        return prob[0], h, c

    def predict(self, x, max_length=50):
        """
        テスト用の予測メソッド
        :param x: 入力文
        :param max_length: 出力文の制御
        :return: 出力文のindex
        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            result = []
            y = [EOS]

            # Embedding Layer
            x = x[::-1]  # 入力を反転したほうが精度が上がる
            emb_x = [self.x_embed(x)]

            # Encoder
            h, c, a = self.encoder(None, None, emb_x)

            # Decoder, Output Layerの計算
            for i in range(max_length):
                prob, h, c = self.generate_one_step(y, h, c, a)  # decoderの計算
                y = self.xp.argmax(prob.data)

                # 出力がEOS信号かcheck。もしそうならbreak
                if y == EOS:
                    break
                result.append(y)
                # 次の入力へのVariable化
                y = self.xp.array([y], dtype=self.xp.int32)

            return result

    def beam_search_predict(self, x, max_length=10, beam_width=3):
        """
        ビームサーチを用いたpredict
        :param x: 入力文
        :param max_length: 出力文の制御
        :param beam_width: ビーム幅
        :return: 出力文のindex
        """
        # import heapq

        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            result = []
            y = [EOS]

            # Embedding Layer
            x = x[::-1]  # 入力を反転したほうが精度が上がる
            emb_x = [self.x_embed(x)]

            # Encoder
            h, c, a = self.encoder(None, None, emb_x)

            # beam search
            heaps = [[] for _ in range(max_length + 1)]
            heaps[0].append((0, y, h, c))  # socre, word, hidden state, cell state
            result_score = 1e8

            # Decoder, Output Layerの計算
            for i in range(max_length):
                heaps[i] = sorted(heaps[i], key=lambda t: t[0])[:beam_width]

                for score, y, h, c in heaps[i]:
                    prob, h, c = self.generate_one_step(y, h, c, a)  # decoderの計算

                    for next_index in self.xp.argsort(prob.data)[::-1]:

                        if prob.data[next_index] < 1e-6:
                            break
                        next_score = score - self.xp.log(prob.data[next_index])

                        if next_score > result_score:
                            break
                        next_word = y + [next_index]
                        next_item = (next_score, next_word, h, c)

                        if next_index == EOS:
                            if next_score < result_score:
                                result = y[1:]  # EOS信号の削除
                                print("result: {}".format(result))
                                result_score = next_score
                        else:
                            heaps[i+1].append(next_item)

            return result
