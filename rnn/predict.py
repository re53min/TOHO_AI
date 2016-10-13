#!/usr/bin/env python
# -*- coding: utf_8 -*-

import cPickle as pickle


def load():
    # Vocablaryの読み込み
    vocab = pickle.load(open("vocab.bin", "rb"))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # Modelの読み込み
    model = pickle.load(open("finalmodel", "rb"))
    n_units = model.embed.W.data.shape[1]

    return vocab, ivocab, model, n_units

def predict():




if __name__ == "__main__":

    # ロード
    vocab, ivocab, model, n_units = load()

    # initialize generator


    predict()
