import matplotlib.pyplot as plt
from chainer.computational_graph import build_computational_graph
from chainer.datasets import get_cifar10, get_mnist, get_cifar100
import numpy as np


def get_dataset(dataset):

    if dataset == "mnist":
        # label 0 ~ 10
        n_class = 10
        # mnistのロード
        train, test = get_mnist(ndim=3)

        # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
        train_dataset, test_dataset = split_dataset(train, test)

    elif dataset == "cifar10":
        # label
        n_class = 10
        # cifar10のロード
        train, test = get_cifar10()

        # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
        train_dataset, test_dataset = split_dataset(train, test)

    elif dataset == "cifar100":
        # label
        n_class = 100
        # cifar100
        train, test = get_cifar100()

        # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
        train_dataset, test_dataset = split_dataset(train, test)

    else:
        raise RuntimeError('Invalid dataset choice.')

    return n_class, train_dataset, test_dataset


def split_dataset(train_data, test_data):
    """
    データセットを入力データと教師データに分割
    :param train_data:
    :param test_data:
    :return: tuple (data, label)
    """
    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # trainデータの分割
    for x, l in train_data:
        train_x.append(x)
        train_y.append(l)

    # testデータの分割
    for x, l in test_data:
        test_x.append(x)
        test_y.append(l)

    return (train_x, train_y), (test_x, test_y)


def plot_loss(data):

    plt.figure(figsize=(8, 6))
    if isinstance(data, tuple):
        for tmp in data:
            plt.plot(range(len(tmp)), tmp)
        plt.legend(['loss', 'accuracy'], loc=1)
    else:
        plt.plot(range(len(data)), data)
        plt.legend(['loss'], loc=1)
    # plt.title("Prediction accuracy.")
    plt.show()


def draw_weight(layer, channel, line):

    width = channel/line

    n1, n2, h, w = layer.W.shape
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.00000005)

    for i in range(n1):
        ax = fig.add_subplot(line, width, i+1, xticks=[], yticks=[])
        ax.imshow(layer.W[i, 0].data, cmap='gray', interpolation='nearest')
    plt.show()


def draw_layer(layer, row, line):

    # plt.title("{}".format(str(label)))
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i, channels in enumerate(layer):
        for j, tmp in enumerate(channels):
            ax = fig.add_subplot(line, row, i*10+j+1, xticks=[], yticks=[])
            ax.imshow(tmp, interpolation='nearest')
    plt.show()


def draw_image(output):

    plt.figure(figsize=(8, 8))

    for i, x in enumerate(output):
        plt.subplot(10, 10, i+1)
        plt.axis("off")
        tmp = ((np.vectorize(clip_img)(x)+1)/2).transpose(1, 2, 0)
        print(tmp)
        plt.imshow(tmp)
    plt.show()


def draw_graph(y, name):

    with open(name.format('graph.dot'), 'w') as o:
        g = build_computational_graph([y])
        o.write(g.dump())


def plot_2winx(data1, data2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(data1)), data1, color='b')
    ax2 = ax1.twinx()
    ax2.plot(range(len(data2)), data2, color='r')
    plt.show()


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))
