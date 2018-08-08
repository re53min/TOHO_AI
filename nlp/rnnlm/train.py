from chainer.optimizers import Adam
from chainer.optimizer import GradientClipping
from chainer import Variable
import numpy as np

import time
import pickle

from nlp.rnnlm.rnnlm import RNNLM
from nlp.utils import make_vocab, mecab_wakati, plot_loss


# x = "メリー！ボブスレーしよう！！"
# y = "オッケー蓮子！！"
# input_sentence = ["メリー", "！", "ボブスレー", "しよ", "う", "！", "！"] + ["<eos>"]
# output_sentence = ["オッケー", "蓮子", "！", "！"] + ["<eos>"]
# x = list(x)
# y = list(y)
tmp_vocab = {}
train_x = []

with open('./dataset/train.pickle', 'rb') as f:
    x = pickle.load(f)

vocab = make_vocab(x)

for row in x:
    train_x.append(np.array([vocab[word] for word in mecab_wakati(row).split()], dtype='int32'))

train_x = np.array(train_x)

print("train_x: {}, vocab: {}".format(len(train_x), len(vocab)))

loss = 0
average_loss = []
epochs = 20
batch_size = 50
num_data = len(train_x)
start_at = time.time()
cur_at = start_at

model = RNNLM(
    vocab_size=len(vocab),
    embed_size=300,
    hidden_size=300,
)

optimizer = Adam()
optimizer.setup(model)
optimizer.add_hook(GradientClipping(5))

for c, i in vocab.items():
    tmp_vocab[i] = c

word = Variable(np.array([vocab.get('私', vocab['UNK'])], dtype='int32'))

# エポックを回す
for epoch in range(0, epochs):

    print('EPOCH: {}/{}'.format(epoch+1, epochs))
    perm = np.random.permutation(num_data)  # ランダムサンプリング

    # ミニバッチ単位で回す
    for idx in range(0, num_data, batch_size):
        # 入力データと出力データをスライス
        batch_x = Variable(train_x[perm[idx: idx + batch_size]])  # if idx + batch_size < num_data else num_data]])]

        # modelの最適化
        model.cleargrads()
        loss = model(batch_x)  # ニューラルネットへの入力
        loss.backward()
        optimizer.update()
        now = time.time()
        print('{}/{}, train_loss = {}, time = {:.2f}'.format(
            idx, len(train_x), loss.data, now-cur_at))
        average_loss.append(loss.data)
        cur_at = now

    print('私', end='')
    for index in model.predict(word):
        if index == vocab['EOS']:
            print()
        else:
            print(tmp_vocab[index], end='')
    print()

    # モデルデータの保存
    pickle.dump(model, open('rnnlm_50_tmp.model', 'wb'))

# モデルデータの保存
pickle.dump(model, open('rnnlm_50.model', 'wb'))
# lossのplot
plot_loss(average_loss)
