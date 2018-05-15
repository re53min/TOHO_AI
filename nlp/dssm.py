import chainer
import chainer.links as L
import chainer.functions as F
import math
import numpy as np
from chainer.initializers import GlorotUniform
from sklearn.metrics.pairwise import cosine_similarity


class DSSM(chainer.Chain):
    def __init__(self, n_hidden, n_output):
        self.n_hidden = n_hidden
        self.n_output = n_output

        super(DSSM, self).__init__(
            # query
            query_embed=L.EmbedID(None, self.n_hidden*10),
            query_l1=L.Linear(None, self.n_hidden),
            query_l2=L.Linear(None, self.n_output),

            # document
            doc_embed=L.EmbedID(None, self.n_hidden*10),
            doc_l1=L.Linear(None, self.n_hidden),
            doc_l2=L.Linear(None, self.n_output)
        )
        for param in self.params():
            param.data[...] = GlorotUniform()

    def __call__(self, x):
        query = self.query_layer(x)

        return

    def query_layer(self, x):

        embed = self.query_embed(x)
        h1 = F.tanh(self.query_l1(embed))
        h2 = F.tanh(self.query_l2(h1))

        return h2

    def doc_layer(self, x):

        embed = self.doc_embed(x)
        h1 = F.tanh(self.doc_l1(embed))
        h2 = F.tanh(self.doc_l2(h1))

        return h2

    def cosine_similarity(self, v1, v2):

        return cosine_similarity(v1, v2)
