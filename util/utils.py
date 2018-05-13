import codecs
import io
import pickle
import sys

import numpy as np
from util.mecab import mecab_wakati

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_file(file_path):
    return codecs.open(file_path, 'rb', 'utf_8_sig').read()


def load_model(model, vocab):
    # vocabularyの読み込み
    vocab = pickle.load(open(vocab, "rb"))
    ivocab = {}

    for c, i in vocab.items():
        ivocab[i] = c
    # modelの読み込み
    model = pickle.load(open(model, "rb"))

    return vocab, ivocab, model


def make_vocab_dict(sentences, mecab=True):
    vocab = {}  # Word ID
    # 分かち書き処理
    words = mecab_wakati(sentence=sentences).split(' ') if mecab else list(sentences)
    dataset = np.ndarray((len(words),), dtype=np.int32)  # 全word分のndarrayの作成

    # 単語辞書登録
    for i, word in enumerate(words):
        # wordがvocabの中に登録されていなかったら新たに追加
        if word not in vocab:
            vocab[word] = len(vocab)
        # デーアセットにwordを登録
        dataset[i] = vocab[word]

    print("corpus size: ", len(sentences))
    print("vocabulary size: ", len(vocab))

    return dataset, vocab


if __name__ == "__main__":
    file_path = '..\\dataset\\corpus.txt'
    keywords = []

    dataset, vocab = make_vocab_dict(load_file(file_path))
    print(vocab)
