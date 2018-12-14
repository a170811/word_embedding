
"""
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
"""


###convolotion
"""
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
