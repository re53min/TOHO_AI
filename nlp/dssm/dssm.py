import chainer
import chainer.links as L
import chainer.functions as F
import math
import numpy as np
from chainer.initializers import GlorotUniform
from sklearn.metrics.pairwise import cosine_similarity

UNK = 0
EOS = 1


class DSSM(chainer.Chain):
    def __init__(self, n_vocab, n_hidden, n_output):
        super(DSSM, self).__init__()
        with self.init_scope():
            # query
            self.query_embed = L.EmbedID(n_vocab, n_hidden)
            self.query = L.NStepLSTM(n_layers=2, in_size=n_hidden, out_size=n_output, dropout=0.3)

            # document
            self.doc_embed = L.EmbedID(n_vocab, n_hidden)
            self.doc = L.NStepLSTM(n_layers=2, in_size=n_hidden, out_size=n_output, dropout=0.3)

        for param in self.params():
            param.data[...] = GlorotUniform()

    def __call__(self, x, y):

        batch_size = len(x)

        query = self.query_layer(x)
        doc = self.doc_layer(x)
        cos = cosine_similarity(query, doc)
        loss = F.softmax_cross_entropy(query, doc)

        return cos, loss

    def query_layer(self, x):

        embed_x = [self.query_embed(tmp) for tmp in x]
        h, c, o = self.query(None, None, embed_x)

        return h

    def doc_layer(self, x):

        embed_x = [self.doc_embed(tmp) for tmp in x]
        h, c, o = self.doc(None, None, embed_x)

        return h
