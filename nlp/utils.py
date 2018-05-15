import MeCab
import matplotlib.pyplot as plt


def mecab_wakati(sentence):

    tagger = MeCab.Tagger('-Owakati')
    result = tagger.parse(sentence)

    return result


def make_vocab(words, mecab=True):

    vocab = {
        'UNK': 0,
        'EOS': 1,
    }
    tmp = []
    for w in words:
        tmp.append(mecab_wakati(sentence=w).split(' ') if mecab else list(w))
    # dataset = np.ndarray((len(words),), dtype='int32')

    for w in tmp:
        for i, word in enumerate(w):
            if word not in vocab:
                vocab[word] = len(vocab)
            # dataset[i] = vocab[word]

    return vocab  # dataset, vocab


def plot_loss(data):

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(data)), data)
    plt.legend(["train_acc"], loc=1)
    # plt.title("Prediction accuracy.")
    plt.plot()
    plt.savefig('loss.png')
    plt.savefig('loss.pdf')

    return
