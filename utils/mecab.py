#!/usr/bin/env python
# -*- coding: utf_8 -*-

import codecs
import MeCab
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout)


# 分かち書き処理
def mecab_wakati(sentence):
    """
    MeCabで分かち書き
    :param sentence:
    :return: 分かち書きlist
    """
    # MeCabの設定
    tagger = MeCab.Tagger('-Owakati')
    encoded_text = sentence.encode('utf-8', 'ignore')
    result = tagger.parse(encoded_text).decode('utf-8')

    return result


if __name__ == "__main__":

    sentences = [u"限りなく小さい世界には妖怪が住んでいた……", u"出だしはこれで決まりね"]
    keywords = []

    for sentence in sentences:
        keywords.append(mecab_wakati(sentence=sentence))

    for word in keywords:
        print(word)

