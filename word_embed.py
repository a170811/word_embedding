import tensorflow as tf
from tqdm import tqdm
from preprocess import Preprocessing
import os
import urllib.request

url = 'http://mattmahoney.net/dc/text8.zip' 
data_path = 'text8.zip' 

##load data
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed
with zipfile.ZipFile('./text8.zip') as f :
    text_words = f.read(f.namelist()[0]).lower().decode('utf-8').split()

data = Preprocessing( text_words , min_occurence = 0.1 )

#Parameters

# Network Parameters
n_hidden_1 = 256 
n_hidden_2 = 256 
num_input = 26
windows = 10
dimension = 100
dropout = 0.75

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_input])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

##feedforword network
# Store layers weight & bias
f_weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, dimension]))
}
f_biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([dimension]))
}

# Create model
def neural_net(x , f_weight , f_biases ):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, f_weights['h1']), f_biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, f_weights['h2']), f_biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, f_weights['out']) + f_biases['out']
    return out_layer

##Convolution neural network

# Store layers weight & bias
c_weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([window , dimension,1 , 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
}

c_biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([1024])),
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.sigmoid(x)

def maxpool2d(x, k=dimension-window+1):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, dimension, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Construct feedforword network
center_f = neural_net(X , f_weight , f_biases)
center_c = conv_net(Y , c_weight , c_biases , keep_prob)
"""
#testing data
text_words = ['moment', 'homeless', 'disable', 'bore', 'frustrate', 'apple', 'milk', 'is', 'good', 'to', 'drink', 'delicious']

data_obj = Preprocessing( text_words , min_occurrence = 0.1 )
print( 'text_words = ' , text_words )
print( 'vocabulary size = ' , data_obj.vocabulary_size )
print( 'data : \n' , data_obj.data )
print( 'data_base_form : \n' , data_obj.base )
print( 'tri_tram : \n' , data_obj.text_gram )
print( 'prefix : \n' , data_obj.prefix )
print( 'suffix : \n' , data_obj.suffix )
"""
