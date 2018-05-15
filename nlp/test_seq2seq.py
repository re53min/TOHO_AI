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

    with open('./json/speaker.pickle', 'rb') as f:
        x = pickle.load(f)
    with open('./json/response.pickle', 'rb') as f:
        y = pickle.load(f)

    vocab = make_vocab(x+y)
    tmp_vocab = {}
    for c, i in vocab.items():
        tmp_vocab[i] = c

    with open("./attention_seq2seq.model", mode='rb') as f:
        model = pickle.load(f)

    while True:
        s = input()
        test_input = Variable(np.array(
            [vocab.get(word, vocab['UNK']) for word in mecab_wakati(s).split()], dtype='int32'
        ))

        print('入力-> {}'.format(s))
        print('出力-> ', end="")
        for index in model.predict(test_input):
            print(tmp_vocab[index], end='')
        print()


if __name__ == "__main__":
    main()
