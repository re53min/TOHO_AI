from chainer.optimizers import Adam
from chainer.optimizer import GradientClipping
from chainer import Variable, serializers
import numpy as np

import time
import pickle

from nlp.seq2seq.seq2seq import Seq2Seq
from nlp.utils import make_vocab, mecab_wakati, plot_loss

# x = "メリー！ボブスレーしよう！！"
# y = "オッケー蓮子！！"
# input_sentence = ["メリー", "！", "ボブスレー", "しよ", "う", "！", "！"] + ["<eos>"]
# output_sentence = ["オッケー", "蓮子", "！", "！"] + ["<eos>"]
# x = list(x)
# y = list(y)
tmp_vocab = {}
train_x = []
train_y = []

with open('./json/speaker.pickle', 'rb') as f:
    x = pickle.load(f)
with open('./json/response.pickle', 'rb') as f:
    y = pickle.load(f)

# train_set, vocab = make_vocab(x+y)
vocab = make_vocab(x+y)

for speaker, utterance in zip(x, y):
    train_x.append(np.array([vocab[word] for word in reversed(mecab_wakati(speaker))], dtype='int32'))
    train_y.append(np.array([vocab[word] for word in mecab_wakati(utterance)], dtype='int32'))

train_x = np.array(train_x)
train_y = np.array(train_y)

print("train_x: {}, train_y: {}, vocab: {}".format(len(train_x), len(train_y), len(vocab)))

loss = 0
average_loss = []
accuracy_list = []
epochs = 50
batch_size = 128
num_data = len(train_x)

model = Seq2Seq(
    vocab_size=len(vocab),
    embed_size=512,
    hidden_size=512,
)

optimizer = Adam()
optimizer.setup(model)
optimizer.add_hook(GradientClipping(5))

for c, i in vocab.items():
    tmp_vocab[i] = c

# timer
start_at = time.time()
cur_at = start_at

# エポックを回す
for epoch in range(0, epochs):

    print('EPOCH: {}/{}'.format(epoch+1, epochs))
    perm = np.random.permutation(num_data)  # ランダムサンプリング

    # ミニバッチ単位で回す
    for idx in range(0, 1000, batch_size):
        # 入力データと出力データをスライス
        batch_x = Variable(train_x[perm[idx: idx + batch_size]])  # if idx + batch_size < num_data else num_data]])]
        batch_y = Variable(train_y[perm[idx: idx + batch_size]])  # if idx + batch_size < num_data else num_data]])]

        # modelの最適化
        model.cleargrads()
        loss, accuracy = model(batch_x, batch_y)  # ニューラルネットへの入力
        loss.backward()
        optimizer.update()
        now = time.time()
        print('{}/{}, train_loss = {}, accuracy = {}, time = {:.2f}'.format(
             idx, num_data, loss.data, accuracy.data, now-cur_at))
        average_loss.append(loss.data)
        accuracy_list.append(accuracy.data)
        cur_at = now

    # 1エポック終わる毎に適当な入力を与えて確認
    for tmp in perm[1: 10]:
        # print('入力-> {}'.format(''.join(x[tmp])))
        # print('出力-> ', end='')

        test_x = Variable(train_x[tmp])

        for index in model.beam_search_predict(test_x):
            print(tmp_vocab[index], end='')
        # print(" (正解: {})".format(''.join(y[tmp])))
        print()

# モデルデータの保存
serializers.save_npz('./seq2seq.npz', model)
# lossのplot
plot_loss((average_loss, accuracy_list))
