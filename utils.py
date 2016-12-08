#!/usr/bin/env python
# -*- coding: utf_8 -*-

import codecs
import numpy as np

from mecab import mecab_wakati


def make_vocab_dict(file_path, mecab=True):
    vocab = {}  # Word ID
    with codecs.open(file_path, 'r', 'utf-8') as sentences:  # .replace('\r\n', '<eos>')  # textの読み込み
        sentences = sentences.read() + ["eos"]
        # 分かち書き処理
        if mecab:
            words = mecab_wakati(sentence=sentences)  # .replace(u'\r', u'eos').split(" ")
        else:
            words = list(sentences)
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
            # print(str(word))
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print("corpus size: ", len(words))
    print("vocabulary size: ", len(vocab))

    return dataset, vocab

if __name__ == "__main__":

    sentences = [u"限りなく小さい世界には妖怪が住んでいた……", u"出だしはこれで決まりね"]
    keywords = []

    for sentence in sentences:
        keywords.append(mecab_wakati(sentence=sentence))
