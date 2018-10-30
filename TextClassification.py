import tensorflow as tf
from tensorflow import keras

import numpy as np

#download imdb dataset
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0]) #prints words in 0th review, each word as an integer, associated w dictionary of words

print("len(train_data[0]), len(train_data[1]): ",len(train_data[0]), len(train_data[1])) #diff reviews are diff lengths, so will have to resolve this later

'''---------------------------------------
create a fn to query a dictionary object that contains the integer to string mapping'''
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

'''--------------------------------------'''

print(decode_review(train_data[0]))

'''
pad data so all reviews are same length of 256
'''
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print('len(train_data[0]), len(train_data[1])', len(train_data[0]), len(train_data[1]))
print('padded review[0]: \n',train_data[0])

'''

--------Build the Model -------------

two main questions:
- how many layers to use in the model?
- how many *hidden units* to use for each layer?

'''
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())