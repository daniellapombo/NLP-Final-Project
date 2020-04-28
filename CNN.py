

import pandas as pd
import numpy as np
import re

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

import matplotlib.pyplot as plt
from keras import layers
from keras.engine.sequential import Sequential
from keras.backend.tensorflow_backend import one_hot


# set parameters:

#size of vocab, rounded up
max_features = 7000
#total size of data, rounded up
maxlen = 13000

batch_size = 50
embedding_dims = 50

filters = 200
hidden_dims = 250
kernel_size = 2
epochs = 2





'''
Import data sets and assign corresponding labels
'teen_processed.csv' - reddit teen text
'genx_processed.csv' - reddit GenX text
'''
def import_txtF():
    teen = pd.read_csv('./data/teen_processed.csv', header=[0])
    tn_label = np.zeros((teen.shape[0], 1), dtype=int)
    gX = pd.read_csv('./data/genx_processed.csv', header=[0])
    gX_label = np.ones((gX.shape[0], 1), dtype=int)

    return teen, tn_label, gX, gX_label




'''
Splits dataset up and labels data accordingly
Removes unwanted characters
Resets index and shuffle
Returns split data
'''
def rdttxt_processing():
    tn, tn_l, genX, genX_l = import_txtF()

    #Create dataset (combining all data into one data set of text and labels)
    dat = pd.concat([tn, genX])
    dat = dat.drop(dat.columns[[0]], axis=1)
    dat['labels'] = np.concatenate([tn_l, genX_l])

    #Clean data set of unwanted characters
    dat.replace(to_replace=r'[0-9"\*><\',]',
                value=' ', regex=True, inplace=True)

    #Shuffle data set
    dat = dat.sample(frac=1).reset_index(drop=True)
    doc = dat.iloc[:, 0]

    train_txt, test_txt, train_label, test_label = train_test_split(
        doc, dat['labels'], test_size=.2, train_size=.8, shuffle=True)
    return train_txt, test_txt, train_label, test_label



'''
CNN model

Data is first tokenized, and then sequences are created from tokens

model has 

'''

def CNN(trn, trn_label, test, test_label):

    tokenizer = Tokenizer(num_words=7000)
    
    trn = tokenizer.texts_to_sequences(trn)
    test = tokenizer.texts_to_sequences(test)
    
    vocab_size = len(tokenizer.word_index) + 1
    trn  = pad_sequences(trn, padding='post', maxlen=maxlen)
    test = pad_sequences(test, padding='post', maxlen=maxlen)

    # print('x_train shape:', trn.shape)
    # print('x_test shape:', test.shape)

    model = Sequential()

    #Maps vocab into embedding_dims dimensions
    #input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    # Length of input sequences, when it is constant.
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    # Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
    # float between 0 and 1. Fraction of the input units to drop.
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
    #Average pooling for temporal data.
    # https://keras.io/layers/pooling/
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    history = model.fit(trn, trn_label,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test, test_label))
    model.summary()
    loss, accuracy = model.evaluate(trn, trn_label, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(test, test_label, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



def main():
    trn, test, trn_label, tst_label = rdttxt_processing()
    CNN(trn, trn_label, test, tst_label)
    
  


main()


