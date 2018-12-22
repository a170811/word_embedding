# -*- coding: utf-8 -*-

import argparse
import pickle
import os
import random
import math

import tensorflow as tf
import numpy as np


def next_batch(batch_size, num_sample, dict_size):

    data_index = 0

    while data_index < len(data):

        end = data_index + batch_size
        if end > len(data):
            end = len(data)
            batch_size = len(data) - data_index

        batch_f = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_c = np.ndarray(shape=(batch_size, context_len), dtype=np.int32)

        for idx in range(data_index, end):
            batch_f[idx%batch_size] = data[idx][0]
            batch_c[idx%batch_size] = data[idx][1]

        neg_samples = [random.sample([j for j in range(dict_size) if j != i], num_sample) for i in batch_f]
        data_index += batch_size

        yield batch_f, batch_c, np.array(neg_samples)

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x , W, strides = [1, strides, 1, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return tf.sigmoid(x)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(conv1, ksize = [1, k, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')

if '__main__' == __name__:


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory path')
    parser.add_argument('--e_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--slide_window', type=int, default=5, help='sliding window')
    parser.add_argument('--n_negs', type=int, default=64, help='number of negative samples')
    #parser.add_argument('--n_negs', type=int, default=100, help='number of negative samples')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--temperature', type=int, default=100, help='temperature parameter magnify the influence of simility function')
    parser.add_argument('--mb', type=int, default=128, help='mini-batch size')
    #parser.add_argument('--mb', type=int, default=1024, help='mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=0.02, help='learning rate')

    args =  parser.parse_args()

    data = pickle.load(open(os.path.join(args.data_dir, 'train.dat'), 'rb'))
    hash_matrix = pickle.load(open(os.path.join(args.data_dir, 'hash_matrix.dat'), 'rb'))
    hash_matrix = np.asarray(hash_matrix)

    if not data:
        raise Exception('data loading error')
    dict_size = len(hash_matrix)
    feature_len = len(hash_matrix[0])
    context_len = len(data[0][1])
    assert context_len >= args.slide_window
    #valid_size = 16
    #valid_window = 100
    #valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    #valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_windowo), valid_size//2))

    graph = tf.Graph()

    with graph.as_default():

    ###fully connexcted network
        fc_input = tf.placeholder(tf.int64, shape = [None])
        negative_input = tf.placeholder(tf.int64, shape = [None, args.n_negs])
        #valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

        lookup_f = tf.cast(hash_matrix, tf.float32)
        fc1_w = tf.Variable(tf.random_normal(shape = (feature_len, 128)))
        ##fc1_w = tf.Variable(tf.truncated_normal(shape = (feature_len, 128), mean = 0,
        #        stddev = 1.0 / math.sqrt(feature_len)))
        fc1_b = tf.Variable(tf.zeros(128) + 1)
        fc11 = tf.matmul(lookup_f, fc1_w) + fc1_b
        fc1 = tf.sigmoid(fc11)

        fc2_w = tf.Variable(tf.random_normal(shape = (128, args.e_dim)))
        ##fc2_w = tf.Variable(tf.truncated_normal(shape = (128, args.e_dim), mean = 0,
        #        stddev = 1.0 / math.sqrt(feature_len)))
        fc2_b = tf.Variable(tf.zeros(args.e_dim) + 1)
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        #vocab * 100
        fc2 = tf.sigmoid(fc2)
        embedding_norm = fc2 / tf.sqrt(tf.reduce_sum(tf.square(fc2), 1, keepdims=True))

        #128 * 100
        X_embed = tf.nn.embedding_lookup(embedding_norm, fc_input)
        #negative samples
        negative_samples = tf.reduce_sum(tf.nn.embedding_lookup(embedding_norm,
                negative_input), 1)

    ###convolutional network
        conv_input = tf.placeholder(tf.int32, shape = [None, context_len])

        lookup_c = tf.nn.embedding_lookup(hash_matrix, conv_input)
        lookup_c = tf.cast(lookup_c, tf.float32)
        x = tf.reshape(lookup_c, shape = [-1, context_len, feature_len, 1])
        conv1_w = tf.Variable(tf.random_normal([args.slide_window, feature_len, 1,
                args.e_dim]))
        conv1_b = tf.Variable(tf.random_normal([args.e_dim]))
        conv1 = conv2d(x, conv1_w, conv1_b, 1)
        #k = context_len - slie_window + 1
        #iff convolution stride = 1
        conv1 = maxpool2d(conv1, context_len - args.slide_window + 1)
        conv1 = tf.reshape(conv1, [-1, 100])
        conv1_norm = conv1 / tf.sqrt(tf.reduce_sum(tf.square(conv1)))

    ###cosine_similarity
        #def _cos_sim(a, b):
        #    return tf.reduce_sum(tf.multiply(a, b), 1, keepdims = True)
        #gradient = _cos_sim(X_embed, conv1_norm) - _cos_sim(negative_samples, conv1_norm)

        x = tf.reduce_sum(tf.multiply(X_embed, conv1_norm), 1, keepdims = True)
        y = tf.reduce_sum(tf.multiply(negative_samples, conv1_norm), 1, keepdims = True)
        gradient = x - y

        #loss = tf.log(1 + tf.exp(-args.temperature * gradient))
        loss = tf.reduce_mean(tf.log(1 + tf.exp(-args.temperature * gradient)))

        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        print('Initialized')
        average_loss = 0

        for epoch in range(1, args.epoch + 1):
            print('epoch', epoch)
            print()
            for i, (batch_data_f, batch_data_c, neg_data) in enumerate(next_batch(args.mb,
                    args.n_negs, dict_size)):

                if i % 2 == 0:
                    average_loss /= 2
                    print('Average loss at batch {} is {}'.format(i, average_loss))
                feed_dict = {fc_input: batch_data_f, conv_input: batch_data_c,
                        negative_input: neg_data}

                x, c, n = sess.run([fc11, fc1_b, negative_samples], feed_dict = feed_dict)

                print(x[:3])
                _, l = sess.run([optimizer, loss], feed_dict = feed_dict)
                average_loss += l






