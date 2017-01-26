#!/usr/bin/env python
# -*- coding: utf_8 -*-

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)


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
        plot_tsne(layer.w.date)
    elif type == "pca":
        print("pca")


def plot_tsne(vec):

    # scikt-learnのt-sneによる次元削減と可視化
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    vec_tsne = tsne.fit_transform(vec)
    t0 = time()

    # plot_embedding(vec_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))


def plot_embedding(vec, title=None):

    plt.figure()
    ax = plt.subplot(111)

    # for i in range():


    if title is not None:
        plt.title(title)

    plt.savefig(title+".png")


if __name__ == "__main__":

    plot('t-sne')
