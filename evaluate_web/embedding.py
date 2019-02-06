import pickle
import numpy as np

class Embedding():
    def __init__(self, embedding_dir, dictionary_dir):
        self._embedding = pickle.load(open(embedding_dir, 'rb'))
        self._vocab = pickle.load(open(dictionary_dir, 'rb'))
        self._r_vocab = {idx: word for word, idx in self._vocab.items()}

    def get_web(self, word):
        return self._embedding[self._vocab[word]]

    def get_word(self, vector):
        dot = np.sum(np.multiply(vector, self._embedding), axis = 1).argmax()
        return self._r_vocab[dot]

    def nearest(self, vector, n):
        dot = np.sum(-np.multiply(vector, self._embedding), axis = 1).argsort()
        return [self._r_vocab[i] for i in dot[1:n+1]]

    def exist(self, word):
        return word in self._vocab


if '__main__' == __name__:
    e = Embedding('./../embedding.dat', './../dictionary.dat')
    print('{} \'s embedding is {}'.format('good', e.get_web('good')))
    print('{} \'s nearest 8 words is {}'.format('good', e.nearest(e.get_web('good'), 8)))

