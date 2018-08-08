#!/usr/bin/env python
# -*- coding: utf_8 -*-

import io
import sys
import pickle
from nlp.utils import mecab_wakati, make_vocab

import numpy as np
from chainer import Variable

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():

    with open('./dataset/train.pickle', 'rb') as f:
        x = pickle.load(f)

    vocab = make_vocab(x)
    tmp_vocab = {}
    for c, i in vocab.items():
        tmp_vocab[i] = c

    with open("./rnnlm_50.model", mode='rb') as f:
        model = pickle.load(f)

    word = 'EOS'
    in_x = Variable(np.array([vocab.get(word, vocab['UNK'])], dtype='int32'))

    for index in model.predict(in_x, max_length=1000):
        if index == vocab['EOS']:
            print()
        else:
            print(tmp_vocab[index], end='')
    print()


if __name__ == "__main__":
    main()
