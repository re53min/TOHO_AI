# -*- coding: utf_8 -*-
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, ChainList, initializers


class BottleNeckA(Chain):

    def __init__(self, n_in, ch, n_out, stride=2, groups=1, nobias=True):
        super(BottleNeckA, self).__init__()
        W = initializers.HeNormal()

        with self.init_scope():
            # Convolution Layer
            self.conv1 = L.Convolution2D(n_in, ch, ksize=1, stride=stride, pad=0, initialW=W, nobias=nobias)
            self.conv2 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1, initialW=W, nobias=nobias, groups=groups)
            self.conv3 = L.Convolution2D(ch, n_out, ksize=1, stride=1, pad=0, initialW=W, nobias=nobias)

            # BatchNormalization Layer
            self.bn1 = L.BatchNormalization(ch)
            self.bn2 = L.BatchNormalization(ch)
            self.bn3 = L.BatchNormalization(n_out)

            # Shortcut Connection
            self.conv4 = L.Convolution2D(n_in, n_out, ksize=1, stride=stride, pad=0, initialW=W, nobias=nobias)
            self.bn4 = L.BatchNormalization(n_out)

    def __call__(self, x):

        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = F.relu(self.bn3(self.conv3(h1)))
        # Shortcut Connection
        h2 = F.relu(self.bn4(self.conv4(x)))

        return F.relu((h1 + h2))


class BottleNeckB(Chain):

    def __init__(self, n_in, ch, groups=1, nobias=True):
        super(BottleNeckB, self).__init__()
        W = initializers.HeNormal()

        with self.init_scope():
            # Convolution Layer
            self.conv1 = L.Convolution2D(n_in, ch, ksize=1, stride=1, pad=0, initialW=W, nobias=nobias)
            self.conv2 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1, initialW=W, nobias=nobias, groups=groups)
            self.conv3 = L.Convolution2D(ch, n_in, ksize=1, stride=1, pad=0, initialW=W, nobias=nobias)

            # BatchNormalization Layer
            self.bn1 = L.BatchNormalization(ch)
            self.bn2 = L.BatchNormalization(ch)
            self.bn3 = L.BatchNormalization(n_in)

    def __call__(self, x):

        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))

        # Shortcut Connection
        return F.relu((h1 + x))


class Block(ChainList):

    def __init__(self, layer, n_in, ch, n_out, stride=2, groups=1):
        super(Block, self).__init__()

        self.add_link(BottleNeckA(n_in, ch, n_out, stride, groups))
        for i in range(layer-1):
            self.add_link(BottleNeckB(n_out, ch, groups))

    def __call__(self, x):

        for f in self.children():
            x = f(x)

        return x


class ResNet(Chain):

    def __init__(self, n_out):
        super(ResNet, self).__init__()
        W = initializers.HeNormal()

        with self.init_scope():

            self.conv1 = L.Convolution2D(None, 64, 3, 1, 0, initialW=W)
            self.bn1 = L.BatchNormalization(64)

            # res block (layer, n_in, ch, n_out, stride, groups)
            self.res1 = Block(3, 64, 64, 256, 1)
            self.res2 = Block(4, 256, 128, 512)
            self.res3 = Block(23, 512, 256, 1024)
            self.res4 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(2048, n_out)

    def __call__(self, x, t):

        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), ksize=3, stride=2)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        y = self.fc(h)

        loss = F.softmax_cross_entropy(y, t)

        return loss, F.accuracy(F.softmax(y), t)

    def predict(self, x, t):

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            h = self.bn1(self.conv1(x))
            h = F.max_pooling_2d(F.relu(h), ksize=3, stride=2)
            h = self.res1(h)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = F.average_pooling_2d(h, h.shape[2:])
            y = self.fc(h)

            return F.argmax(F.softmax(y)), F.accuracy(F.softmax(y), t)
