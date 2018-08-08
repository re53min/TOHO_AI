import unicodedata

import MeCab
import re
import matplotlib.pyplot as plt
import numpy as np


def mecab_wakati(sentence):

    tagger = MeCab.Tagger('-Owakati')
    result = tagger.parse(sentence).split(' ')
    result = list(filter(lambda a: a != "\n", result))

    return result


def make_vocab(words, mecab=True):

    # if vocab is None:
    #     vocab = {
    #         'UNK': 0,
    #         'EOS': 1,
    #     }
    # else:
    #     vocab['UNK'] = 0
    #     vocab['EOS'] = 1

    vocab = {'UNK': 0, 'EOS': 1}
    tmp = []
    import gc
    print(len(words))
    for i, w in enumerate(words):
        print(i)
        tmp.append(mecab_wakati(sentence=w) if mecab else list(w))
        del w
        gc.collect()

    # dataset = np.ndarray((len(words),), dtype='int32')
    print("make vocab")
    for w in tmp:
        for i, word in enumerate(w):
            if word not in vocab:
                vocab[word] = len(vocab)
            # dataset[i] = vocab[word]

    return vocab  # dataset, vocab


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


def clean_text(text):

    replaced_text = re.sub(r'<.*>', "", text)  # <>の除去
    replaced_text = re.sub(r'[・※■*◆]', "", replaced_text)  # 図形の除去
    replaced_text = re.sub(r'[【】]', "", replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', "", replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', "", replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', "", replaced_text)  # メンションの除去
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', "", replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', "", replaced_text)  # 全角空白の除去
    replaced_text = re.sub('\u3000', "", replaced_text)
    replaced_text = re.sub(r'[―]', "", replaced_text)

    return unicodedata.normalize('NFKC', replaced_text)


def load_chainer_wordvector(path):

    with open(path, 'r', encoding='utf-8-sig') as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = np.empty((n_vocab, n_units), dtype=np.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            word2index[word] = i
            index2word[i] = word
            w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

    s = np.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))  # normalize

    return w, word2index, index2word
