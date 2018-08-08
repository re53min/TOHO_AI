import pickle
import time

from chainer import Variable
from chainer.optimizer import GradientClipping
from chainer.optimizers import Adam
from chainer.datasets import get_cifar10, get_cifar100, get_mnist

import numpy as np

from vision.resnet.resnet import ResNet
from vision.utils import plot_loss, draw_image, draw_layer, split_dataset, draw_graph

dataset = "cifar10"

if dataset == "mnist":
    # label 0 ~ 10
    n_class = 10
    # mnistのロード
    train, test = get_mnist(ndim=3)

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = np.array(train_dataset[0])
    train_y = np.array(train_dataset[1])
elif dataset == "cifar10":
    # label
    n_class = 10
    # cifar10のロード
    train, test = get_cifar10()

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = np.array(train_dataset[0])
    train_y = np.array(train_dataset[1])
elif dataset == "cifar100":
    # label
    n_class = 100
    # cifar100
    train, test = get_cifar100()

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = np.array(train_dataset[0])
    train_y = np.array(train_dataset[1])
else:
    raise RuntimeError('Invalid dataset choice.')

loss = 0
average_loss = []
accuracy_list = []
n_train_data = len(train)
start_at = time.time()
cur_at = start_at

# ハイパーパラメータ
epochs = 10
batch_size = 100

# モデル構築
model = ResNet(
    n_out=n_class
)

optimizer = Adam()
optimizer.setup(model)
optimizer.add_hook(GradientClipping(5))

# エポックを回す
for epoch in range(0, epochs):

    print('EPOCH: {}/{}'.format(epoch+1, epochs))
    perm = np.random.permutation(n_train_data)  # ランダムサンプリング

    # ミニバッチ単位で回す
    for idx in range(0, n_train_data, batch_size):
        # 入力データと出力データをスライス
        batch_x = Variable(train_x[perm[idx: idx + batch_size]])
        batch_y = Variable(train_y[perm[idx: idx + batch_size]])

        # modelの最適化
        model.cleargrads()
        loss, accuracy = model(batch_x, batch_y)  # ニューラルネットへの入力
        loss.backward()
        optimizer.update()
        now = time.time()
        print('{}/{}, train_loss = {}, time = {:.2f}'.format(
            idx, len(train_x), loss.data, now-cur_at))
        average_loss.append(loss.data)
        accuracy_list.append(accuracy.data)
        cur_at = now

# test
test_x = Variable(np.array(test_dataset[0]))
test_y = Variable(np.array(test_dataset[1]))

# score, test_accuracy = model.predict(test_x, test_y)

# print("Accuracy: {}".format(test_accuracy.data))
# print("test label: {}".format(test_y[test_perm[idx]]))
# print("Classifier: {}".format(score.data))

# モデルデータの保存
# pickle.dump(model, open('ResNet100.model', 'wb'))
# lossのplot
# plot_loss((average_loss, accuracy_list))
# 計算グラフの可視化
# draw_graph(loss)
