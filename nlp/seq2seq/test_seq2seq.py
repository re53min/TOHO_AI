#!/usr/bin/env python
# -*- coding: utf_8 -*-

import io
import sys
import dill
from nlp.utils import mecab_wakati

import numpy as np
from chainer import Variable, serializers
from nlp.seq2seq.seq2seq import Seq2Seq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():

    with open('./word2index.dict', 'rb') as f:
        word2index = dill.load(f)
    with open('./index2word.dict', 'rb') as f:
        index2word = dill.load(f)

    model = Seq2Seq(
        vocab_size=len(word2index),
        embed_size=300,
        hidden_size=300,
    )
    serializers.load_npz('seq2seq.npz', model)

    while True:
        s = input()
        test_input = Variable(
            np.array([word2index.get(word, word2index['UNK']) for word in mecab_wakati(s)], dtype='int32')
        )

        print('入力-> {}'.format(s))
        print('出力-> ', end="")
        for index in model.predict(test_input):
            print(index2word[index], end='')
        print()


if __name__ == "__main__":
    main()
