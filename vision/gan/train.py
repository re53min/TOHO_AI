import pickle
import time

from chainer import Variable
from chainer.optimizer import GradientClipping
from chainer.optimizers import Adam
from chainer.datasets import get_cifar10, get_cifar100, get_mnist
from chainer import cuda
import chainer.functions as F

import numpy as np

from vision.gan.dcgan import Generator, Discriminator
from vision.utils import plot_loss, draw_image, draw_layer, split_dataset, draw_graph

dataset = "cifar10"

# GPU
gpu_device = 0
# uda.get_device(gpu_device).use()
xp = np  # cuda.cupy if gpu_device >= 0 else np

if dataset == "mnist":
    # label 0 ~ 10
    n_class = 10
    # mnistのロード
    train, test = get_mnist(ndim=3)

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = xp.array(train_dataset[0])
    train_y = xp.array(train_dataset[1])
elif dataset == "cifar10":
    # label
    n_class = 10
    # cifar10のロード
    train, test = get_cifar10()

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = xp.array(train_dataset[0])
    train_y = xp.array(train_dataset[1])
elif dataset == "cifar100":
    # label
    n_class = 100
    # cifar100
    train, test = get_cifar100()

    # 本来ならiteratorで回すがわかりやすようにデータとラベルで分割
    train_dataset, test_dataset = split_dataset(train, test)

    train_x = xp.array(train_dataset[0])
    train_y = xp.array(train_dataset[1])
else:
    raise RuntimeError('Invalid dataset choice.')

loss = 0
gen_loss = []
dis_loss = []
n_train_data = len(train_x)

# ハイパーパラメータ
epochs = 1
batch_size = 100
n_hidden = 100

# Generator
generator = Generator(n_hidden=n_hidden)
opt_gen = Adam()
opt_gen.setup(generator)
opt_gen.add_hook(GradientClipping(5))
loss_gen = 0

# Discriminator
discriminator = Discriminator()
opt_dis = Adam()
opt_dis.setup(discriminator)
opt_dis.add_hook(GradientClipping(5))
loss_dis = 0

# time
start_at = time.time()
cur_at = start_at

# エポックを回す
for epoch in range(0, epochs):

    print('EPOCH: {}/{}'.format(epoch+1, epochs))
    perm = xp.random.permutation(n_train_data)  # ランダムサンプリング

    # ミニバッチ単位で回す
    for idx in range(0, 100, batch_size):
        '''
        Generatorの出力y1について、Discriminatorは偽物(=1)と判断するように学習を実施する。
        '''

        # ixput z
        z = Variable(generator.make_hidden(batch_size))
        x = generator(z)
        y1 = discriminator(x)

        # Train Generator
        loss_gen = F.softmax_cross_entropy(y1, Variable(xp.zeros(batch_size, dtype=np.int32)))
        loss_dis = F.softmax_cross_entropy(y1, Variable(xp.ones(batch_size, dtype=np.int32)))

        # Train Discriminator
        batch_x = Variable(train_x[perm[idx: idx + batch_size]])
        y2 = discriminator(batch_x)
        loss_dis += F.softmax_cross_entropy(y2, Variable(xp.zeros(batch_size, dtype=np.int32)))

        # Generatorの最適化
        generator.cleargrads()
        loss_gen.backward()
        opt_gen.update()
        # Discriminatorの最適化
        discriminator.cleargrads()
        loss_dis.backward()
        opt_dis.update()

        now = time.time()
        print('{}/{}, Gen_loss = {}, Dis_loss = {}, time = {:.2f}'.format(
            idx, n_train_data, loss_gen.data, loss_dis.data, now-cur_at))
        gen_loss.append(loss_gen.data)
        dis_loss.append(loss_dis.data)
        cur_at = now

    pickle.dump(generator, open('generator_snapshot.model', 'wb'))
    pickle.dump(discriminator, open('discriminator_snapshot.model', 'wb'))

    z = Variable(generator.make_hidden(batch_size))
    x = generator(z)
    draw_image(x.data)

# モデルデータの保存
pickle.dump(generator, open('generator.model', 'wb'))
pickle.dump(discriminator, open('discriminator.model', 'wb'))
# lossのplot
plot_loss((gen_loss, dis_loss))
# 計算グラフの可視化
draw_graph(loss_gen, "Generator_{}")
draw_graph(loss_dis, "Discriminator_{}")
