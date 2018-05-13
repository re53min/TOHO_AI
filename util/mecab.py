#!/usr/bin/env python
# -*- coding: utf_8 -*-

import MeCab


# 分かち書き処理
def mecab_wakati(sentence):
    """
    MeCabで分かち書き
    :param sentence:
    :return: 分かち書きlist
    """
    # MeCabの設定
    tagger = MeCab.Tagger('-Owakati')
    result = tagger.parse(sentence)

    return result


if __name__ == "__main__":

    sentences = [u"限りなく小さい世界には妖怪が住んでいた……", u"出だしはこれで決まりね"]
    keywords = []

    for sentence in sentences:
        keywords.append(mecab_wakati(sentence=sentence))

    for word in keywords:
        print(word)

