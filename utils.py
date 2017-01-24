#!/usr/bin/env python
# -*- coding: utf_8 -*-

import codecs
import numpy as np
import pickle
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from mecab import mecab_wakati


def load_file(file_path):
    return codecs.open(file_path, 'r', 'utf_8_sig').read()


def load_model(model, vocab):
    # vocabularyの読み込み
    vocab = pickle.load(open(vocab, "rb"))
    ivocab = {}

    for c, i in vocab.items():
        ivocab[i] = c
    # modelの読み込み
    model = pickle.load(open(model, "rb"))

    return vocab, ivocab, model


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


def plot_loss(plot_loss):

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(plot_loss)), plot_loss)
    plt.legend(["train_acc"], loc=1)
    # plt.title("Prediction accuracy.")
    plt.plot()
    plt.savefig('plot_image\\loss.png')
    plt.savefig('plot_image\\loss.pdf')

    return


def plot(type, layer):

    if type == "t-sne":
        plot_tsne()
    elif type == "pca":
        print("pca")


def plot_tsne(vec):

    # scikt-learnのt-sneによる次元削減と可視化
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    vec_tsne = tsne.fit_transform(vec)
    t0 = time()

    plot_embedding(vec_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.savefig(title+".png")


if __name__ == "__main__":
    file_path = 'player1.txt'
    keywords = []

    dataset, vocab = make_vocab_dict(load_file(file_path))
    print(vocab)
