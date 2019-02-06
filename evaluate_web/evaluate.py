from embedding import *

import urllib.request
import os
import numpy as np

def rho(predict, label):
    p_mean = np.mean(predict)
    l_mean = np.mean(label)

    return np.dot(predict - p_mean, label - l_mean)/(np.linalg.norm(predict - p_mean) * np.linalg.norm(label - l_mean))


def sim353(embedding, mode = 'all'):

    if mode == 'all':
        link = 'https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1'
    elif mode == 'set1':
        link = 'https://www.dropbox.com/s/opj6uxzh5ov8gha/EN-WS353-SET1.txt?dl=1'
    elif mode == 'set2':
        link = 'https://www.dropbox.com/s/w03734er70wyt5o/EN-WS353-SET2.txt?dl=1'
    else:
        raise Exception('the mode {} doesn\'t exist!'.format(mode))

    file_dir = './sim353_{}.txt'.format(mode)
    if not os.path.isfile(file_dir):
        urllib.request.urlretrieve(link, file_dir)

    with open(file_dir, 'r') as f:
        data = f.read().splitlines()[1:]
        data = [i.split() for i in data]

    not_found = 0
    predict = []
    label = []
    for query in data:
        if embedding.exist(query[0]) and embedding.exist(query[1]):
            predict.append(np.dot(embedding.get_web(query[0]), embedding.get_web(query[1])))
            label.append(float(query[2]))
        else:
            not_found += 1

    return rho(predict, label), not_found, len(data)

if '__main__' == __name__:
    embedding = Embedding('./../model/embedding.dat', './../data/word2idx.dat')

    result = sim353(embedding, 'all')
    print('rho value = {:.3f}, \nand {} data not found \ntotal: {} data'.format(*result))
