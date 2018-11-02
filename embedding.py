import gzip
from io import open

import numpy as np


def read_embedding(embedding_filename, encoding):
    if embedding_filename.endswith(".gz"):
        external_embedding_fp = gzip.open(embedding_filename, 'rb')
    else:
        external_embedding_fp = open(embedding_filename, 'rb')

    def embedding_gen():
        for line in external_embedding_fp:
            fields = line.decode(encoding).strip().split(' ')
            if len(fields) <= 2:
                continue
            token = fields[0]
            vector = [float(i) for i in fields[1:]]
            yield token, vector

    external_embedding = list(embedding_gen())
    external_embedding_fp.close()
    return external_embedding


class ExternalEmbeddingLoader(object):
    def __init__(self, embedding_filename, encoding="utf-8", dtype=np.float32):
        words_and_vectors = read_embedding(embedding_filename, encoding)
        self.dim = len(words_and_vectors[0][1])
        words_and_vectors.insert(0, ("*UNK*", np.array([0] * self.dim)))

        words, vectors = zip(*words_and_vectors)
        self.lookup = {word: idx for idx, word in enumerate(words)}
        self.vectors = np.array(vectors, dtype=dtype)

