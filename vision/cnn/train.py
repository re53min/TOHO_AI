import pickle
import time

import chainer
from chainer import Variable
from chainer.optimizer import GradientClipping
from chainer.optimizers import Adam
from chainer.datasets import get_cifar10, get_cifar100, get_mnist

import numpy as np

from vision.cnn.cnn import CNN
from vision.utils import plot_2winx, draw_image, draw_layer, split_dataset, draw_graph

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
n_hidden = 100
out_channel = 10
epochs = 30
batch_size = 100

# モデル構築
model = CNN(
    n_class=n_class,
    n_hidden=n_hidden,
    out_channel=out_channel,
    nobias=False
)

optimizer = Adam()
optimizer.setup(model)
optimizer.add_hook(GradientClipping(5))

draw_image(train_x[0:100])

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

score, test_accuracy = model.predict(test_x, test_y)

print("Accuracy: {}".format(test_accuracy.data))
# print("test label: {}".format(test_y[test_perm[idx]]))
# print("Classifier: {}".format(score.data))

# モデルデータの保存
pickle.dump(model, open('cnn.model', 'wb'))
# lossのplot
plot_2winx(average_loss, accuracy_list)

# 中間層の重みの可視化
# draw_weight(model.conv1_1, out_channel, line=2)
# 中間層の出力の可視化
# draw_image(test_x.data[0:9])
layers = model.visualize_layer_output(test_x)
for layer in layers:
    draw_layer(layer.data[0:9], 10, 10)

draw_graph(loss)
