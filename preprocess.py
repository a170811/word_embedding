import collections
import zipfile
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from text_utils import *

###change to class form
###testing

class Preprocessing():

    def __init__(self , url = 'http://mattmahoney.net/dc/text8.zip' ,data_path = 'text8.zip' , max_vocabulary_size = 50000 , min_occurrence = 10) :

        if not os.path.exists(data_path):
            print("Downloading the dataset... (It may take some time)")
            filename, _ = urllib.urlretrieve(url, data_path)
            print("Done!")
        # Unzip the dataset file. Text has already been processed
        with zipfile.ZipFile('./text8.zip') as f :
            text_words = f.read(f.namelist()[0]).lower().decode('utf-8').split()

        #testing data
        #text_words = ['moment', 'homeless', 'disable', 'bore', 'frustrate', 'apple', 'milk', 'is', 'good', 'to', 'drink', 'delicious']

        #inflection to base
        dict2base_word = load_base_dict()
        text_base = word_base( text_words , dict2base_word )


        ##1. self.vocabulary_size
        # Build the dictionary and replace rare words with UNK token
        count = [('UNK', -1)]
        # Retrieve the most common words
        count.extend(collections.Counter(text_words+text_base).most_common(max_vocabulary_size - 1)) #less to many
        # Remove samples with less than 'min_occurrence' occurrences
        for i in range(len(count) - 1, 0, -1):
            if count[i][1] < min_occurrence*2:
                count.pop(i)
            else:
                # The collection is ordered, so stop when 'min_occurrence' is reached
                break
        # Compute the vocabulary size
        self.vocabulary_size = len(count)

        ##2. self.word2id and self.id2word
        # Assign an id to each word
        self.word2id = dict()
        for i, (word, _)in enumerate(count):
            self.word2id[word] = i

        # dictionary:word to id
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        ##3. self.data
        #data to index
        self.data = list()
        for word in text_words:
            # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
            index = self.word2id.get(word, 0)
            self.data.append(index)
        print('data = ' , self.data)

        ##4. self.data_base
        #data_base to index
        self.base = list()
        for word in text_base:
            # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
            index = self.word2id.get(word, 0)
            self.base.append(index)
        print('base = ' , self.base)

        ##4. self.text_gram
        #tri-gram part
        dict2index , _ = load_dict()
        self.text_gram = tri_gram( text_words , dict2index , length = 20 ) #tri_gram(text_words -> list )


        ##5. self.prefix and self.suffix
        #prefix and suffix vector
        self.prefix , self.suffix = word_affix( text_words , 2 )


text_words = ['moment', 'homeless', 'disable', 'bore', 'frustrate', 'apple', 'milk', 'is', 'good', 'to', 'drink', 'delicious']
print(text_words)
test = Preprocessing()
print(test.data)
print(test.base)
print(test.text_gram)
print(test.prefix)
print(test.suffix)

""" backup

# Download a small chunk of Wikipedia articles collection
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = 'text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed
with zipfile.ZipFile('./text8.zip') as f :
    text_words = f.read(f.namelist()[0]).lower().decode('utf-8').split()

###pre-processing


##word embedding

#parameter
max_vocabulary_size = 50000 # Total number of different words in the vocabulary
min_occurrence = 10 # Remove all words that does not appears at least n times

# Build the dictionary and replace rare words with UNK token
count = [('UNK', -1)]
# Retrieve the most common words
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1)) #less to many
# Remove samples with less than 'min_occurrence' occurrences
for i in range(len(count) - 1, 1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached
        break
# Compute the vocabulary size
vocabulary_size = len(count)

# Assign an id to each word
word2id = dict()
for i, (word, _)in enumerate(count):
    word2id[word] = i

data = list()
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
    index = word2id.get(word, 0)
    data.append(index)


id2word = dict(zip(word2id.values(), word2id.keys()))

##n-gram
dict2index , _ = load_dict()
text_gram = tri_gram( text_words , dict2index , length = 20 ) #tri_gram(text_words -> list )


##inflection to base
dict2base_word = load_base_dict()
text_base = word_base( text_words , dict2base_word )

##affix
prefix , suffix = word_affix( text_words , 2 )
print(prefix[1:10])
print(suffix[1:10])

"""
"""
        #backup
        #infle whether to vec?
        count = [('UNK', -1)]
        count.extend(collections.Counter(text_base).most_common(max_vocabulary_size - 1)) #less to many
        for i in range(len(count) - 1, 1, -1):
            print(i)
            if count[i][1] < min_occurrence:
                count.pop(i)
            else:
                # The collection is ordered, so stop when 'min_occurrence' is reached
                break

        self.word2id = dict()
        for i, (word, _)in enumerate(count):
            self.word2id[word] = i

        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        self.infl_data = list()
        for word in text_base:
            # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
            index = self.word2id.get(word, 0)
            self.infl_data.append(index)
"""
