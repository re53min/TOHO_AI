#!/usr/bin/env python
# -*- coding: utf_8 -*-

from __future__ import print_function
import codecs
import numpy as np

from mecab import mecab_wakati


def load_file(file_path):
    return codecs.open(file_path, 'r', 'utf-8').read()


def make_vocab_dict(sentences, mecab=True):
    vocab = {}  # Word ID
    # 分かち書き処理
    words = mecab_wakati(sentence=sentences).split(' ') if mecab else list(sentences)
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print("corpus size: ", len(sentences))
    print("vocabulary size: ", len(vocab))

    return dataset, vocab


if __name__ == "__main__":
    file_path = 'player1.txt'
    keywords = []

    dataset, vocab = make_vocab_dict(load_file(file_path))
    print(vocab)