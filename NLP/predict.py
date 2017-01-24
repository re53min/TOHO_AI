#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function

import codecs
import os
import sys

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np
import six

sys.path.append(os.pardir)
from mecab import mecab_wakati
from utils import load_model, make_vocab_dict, load_file

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


def test_rnnlm(model="rnnlm_model", vocab="rnnlm_vocab.bin", length=10000, sample=0):
    # ロード
    vocab, ivocab, model = load_model(model, vocab)
    xp = np
    model.predictor.reset_state()

    # 標準入力
    # print 'Please your typing!!'
    input_text = u"start"
    if isinstance(input_text, six.binary_type):
        input_text = input_text.decode('utf-8')
    # 入力された文字がvocabの中に含まれていたらprev_wordの生成
    if input_text in vocab:
        prev_word = chainer.Variable(xp.array([vocab[input_text]], xp.int32))
    else:
        print('Error: Unfortunately ' + input_text + ' is unknown')
        exit()
    # 初めの一文字の出力
    # sys.stdout.write(input_text + ' ')
    prob = F.softmax(model.predictor(prev_word))

    for i in range(length):
        # 次の単語の予測
        prob = F.softmax(model.predictor(prev_word))

        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        # eosタグが予測された場合に読点に置換
        if ivocab[index] == u'eos':
            # break
            sys.stdout.write('\r\n')
        else:
            sys.stdout.write(ivocab[index] + ' ')
        # 次の文字へ
        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')


def test_seq2seq(input_text, model="seq2seq_model", vocab="seq2seq_vocab.bin", length=5000, sample=0):
    # ロード
    output_vocab, ivocab, model = load_model(model, vocab)
    model.reset_state()
    # 入力
    in_set, input_vocab = make_vocab_dict(input_text)
    input_text = input_text.split()

    # テスト用
    for sentence in input_text:
        # test = "メリー！ボブスレーしよう！！"
        inputs = [input_vocab[word] for word in reversed(mecab_wakati(sentence).split())]
        print(u'入力 -> ' + sentence)
        print(u'出力 -> ', end='')
        for index in model.predict(inputs, output_vocab):
            print(ivocab[index], end='')
        print()


if __name__ == "__main__":
    # test_rnnlm()
    test_seq2seq(input_text=load_file('train_data\\test_player.txt'))
