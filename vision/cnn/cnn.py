# -*- coding: utf_8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain


class CNN(Chain):

    def __init__(self, n_class, n_hidden, in_channel=None, out_channel=None, nobias=True):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channel, out_channel, ksize=2, stride=1, nobias=nobias)
            self.conv2 = L.Convolution2D(out_channel, out_channel, ksize=3, stride=1, nobias=nobias)
            self.conv3 = L.Convolution2D(in_channel, out_channel, ksize=2, stride=1, nobias=nobias)

            self.fc1 = L.Linear(None, n_hidden, nobias=nobias)
            self.fc2 = L.Linear(n_hidden, n_class, nobias=nobias)

    def __call__(self, x, y, drop=True):

        conv1 = F.relu(self.conv1(x))
        pool1 = F.max_pooling_2d(conv1, ksize=2, stride=2)
        conv2 = F.relu(self.conv2(pool1))
        pool2 = F.max_pooling_2d(conv2, ksize=2, stride=2)
        conv3 = F.relu(self.conv3(pool2))
        pool3 = F.max_pooling_2d(conv3, ksize=2, stride=2)

        fc1 = F.dropout(F.relu(self.fc1(pool3)))
        fc2 = self.fc2(fc1)

        loss = F.softmax_cross_entropy(fc2, y)
        accuracy = F.accuracy(F.softmax(fc2), y)

        return loss, accuracy

    def predict(self, x, y):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            conv1 = F.relu(self.conv1(x))
            pool1 = F.max_pooling_2d(conv1, ksize=2, stride=2)
            conv2 = F.relu(self.conv2(pool1))
            pool2 = F.max_pooling_2d(conv2, ksize=2, stride=2)
            conv3 = F.relu(self.conv3(pool2))
            pool3 = F.max_pooling_2d(conv3, ksize=2, stride=2)

            fc1 = F.relu(self.fc1(pool3))
            fc2 = F.softmax(self.fc2(fc1))

        return F.argmax(fc2), F.accuracy(fc2, y)

    def visualize_layer_output(self, x):

        layer = []
        conv1 = F.relu(self.conv1(x))
        pool1 = F.max_pooling_2d(conv1, ksize=2, stride=2)
        conv2 = F.relu(self.conv2(pool1))
        pool2 = F.max_pooling_2d(conv2, ksize=2, stride=2)
        conv3 = F.relu(self.conv3(pool2))
        pool3 = F.max_pooling_2d(conv3, ksize=2, stride=2)

        layer.append(conv1)
        layer.append(pool1)
        layer.append(conv2)
        layer.append(pool2)
        layer.append(conv3)
        layer.append(pool3)

        return layer
