from chainer.optimizers import Adam
from chainer.optimizer import GradientClipping
from chainer import Variable
import numpy as np

import time
import pickle

from nlp.seq2seq import Seq2Seq
from nlp.utils import make_vocab, mecab_wakati
from util.visualizer import plot_loss


# x = "メリー！ボブスレーしよう！！"
# y = "オッケー蓮子！！"
# input_sentence = ["メリー", "！", "ボブスレー", "しよ", "う", "！", "！"] + ["<eos>"]
# output_sentence = ["オッケー", "蓮子", "！", "！"] + ["<eos>"]
# x = list(x)
# y = list(y)
tmp_vocab = {}
train_x = []
train_y = []

with open('./dataset/json/speaker.pickle', 'rb') as f:
    x = pickle.load(f)
with open('./dataset/json/response.pickle', 'rb') as f:
    y = pickle.load(f)

# train_set, vocab = make_vocab(x+y)
vocab = make_vocab(x+y)

for speaker, utterance in zip(x, y):
    train_x.append(np.array([vocab[word] for word in reversed(mecab_wakati(speaker).split())], dtype='int32'))
    train_y.append(np.array([vocab[word] for word in mecab_wakati(utterance).split()], dtype='int32'))

train_x = np.array(train_x)
train_y = np.array(train_y)

print("train_x: {}, train_y: {}, vocab: {}".format(len(train_x), len(train_y), len(vocab)))

loss = 0
average_loss = []
epochs = 20
batch_size = 100
num_data = len(train_x)
start_at = time.time()
cur_at = start_at

model = Seq2Seq(
    vocab_size=len(vocab),
    embed_size=300,
    hidden_size=300,
)

optimizer = Adam()
optimizer.setup(model)
optimizer.add_hook(GradientClipping(5))

for c, i in vocab.items():
    tmp_vocab[i] = c

# エポックを回す
for epoch in range(0, epochs):

    print('EPOCH: {}/{}'.format(epoch+1, epochs))
    perm = np.random.permutation(num_data)  # ランダムサンプリング

    # ミニバッチ単位で回す
    for idx in range(0, num_data, batch_size):
        # 入力データと出力データをスライス
        batch_x = Variable(train_x[perm[idx: idx + batch_size]])  # if idx + batch_size < num_data else num_data]])]
        batch_y = Variable(train_y[perm[idx: idx + batch_size]])  # if idx + batch_size < num_data else num_data]])]

        # modelの最適化
        model.cleargrads()
        loss = model(batch_x, batch_y)  # ニューラルネットへの入力
        loss.backward()
        optimizer.update()
        now = time.time()
        print('{}/{}, train_loss = {}, time = {:.2f}'.format(
             idx, len(train_x), loss.data, now-cur_at))
        average_loss.append(loss.data)
        cur_at = now

    # 1エポック終わる毎に適当な入力を与えて確認
    for tmp in perm[1: 10]:
        print('入力-> {}'.format(''.join(x[tmp])))
        print('出力-> ', end='')

        test_x = Variable(train_x[tmp])

        for index in model.predict(test_x):
            print(tmp_vocab[index], end='')
        print(" (正解: {})".format(''.join(y[tmp])))
        print()

# モデルデータの保存
pickle.dump(model, open('attention_seq2seq.model', 'wb'))
# lossのplot
plot_loss(average_loss)
