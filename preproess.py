# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
from text_utils import *
from tqdm import tqdm 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def skipgram(self, sentence, i):
        dimension = len(sentence[0])
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [[0]*dimension for _ in range(self.window - len(left))] + left + right + [[0]*dimension for _ in range(self.window - len(right))]

    def build(self, filepath, max_vocab=50000):
        print("building vocab...")
        self.wc = {self.unk: 1}
        with codecs.open(filepath, 'r', encoding='utf-8') as file:

            sent_base = set()
            dict2base_word = load_base_dict()

            for line in tqdm(file, ncols = 80):
                line = line.strip()
                if not line:
                    continue
                sent = line.split() 
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1

                sent_base = sent_base | set(word_base(sent, dict2base_word))


        print("")
        #value from large to small
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        self.base2idx = {word: idx for idx, word in enumerate(sent_base, 1)}

        print('constructing tri-gram pair')
        self.tri2idx = make_dict()
        print('constructing affix pair')
        self.affix2idx = load_affix('./prefix.txt', './suffix.txt')

        
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        pickle.dump(self.tri2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        pickle.dump(self.affix2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        data = []
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file, ncols = 80):
                line = line.strip()
                if not line:
                    continue
                sent = []
                line = line.split()

                base = word_base(line, self.base2idx)
                tri = tri_gram(line, self.tri2idx, 12)
                affix = word_affix(line, self.affix2idx)

                for i, word in enumerate(line):
                    if word in self.vocab:
                        sent.append([self.word2idx[word]])
                    else:
                        sent.append([self.word2idx[0]])

                    #word(1)+base(1)+tri(n)+affix(2)
                    sent[i].extend([base[i]]+tri[i]+affix[i])
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((iword, owords))
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus)
