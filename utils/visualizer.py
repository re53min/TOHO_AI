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

    plot_embedding(vec_tsne, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))


def plot_embedding(vec, title=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(vec.shape[0]):
        plt.text(vec[i, 0], vec[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((vec[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [vec[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                vec[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.savefig(title+".png")


if __name__ == "__main__":

    plot('t-sne')
