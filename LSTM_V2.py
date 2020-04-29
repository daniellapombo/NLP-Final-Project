# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17 
import pandas as pd
import numpy as np
import tensorflow as tf
# def vec(doc):
# We are lemmatizing and removing the stopwords and non-alphabetic characters for each line of dialogue 
# return regexp_tokenize(doc, "[\w']+")
#  from nltk.tokenize import regexp_tokenize 

# Preprocessing
from keras.callbacks import EarlyStopping
from pandas.tests.test_downstream import df
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from nltk.tokenize import regexp_tokenize
from keras.layers import Dense, SpatialDropout1D, LSTM
from keras.layers import Embedding

from keras.engine.sequential import Sequential

# set parameters:

# size of vocab, rounded up
#max_features = 7000
# total size of data, rounded up
# maxlen = 13000

# batch_size = 50
# embedding_dims = 50

# filters = 200
# hidden_dims = 250
# kernel_size = 2
# epochs = 5

'''
Import data sets and assign corresponding labels
'teen_processed.csv' - reddit teen text
'genx_processed.csv' - reddit GenX text
'''


def import_txtF():
    teen = pd.read_csv('data/teen_processed.csv', header=[0])
    tn_label = np.zeros((teen.shape[0], 1), dtype=int)
    gX = pd.read_csv('data/genx_processed.csv', header=[0])
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

    # Create dataset (combining all data into one data set of text and labels)
    dat = pd.concat([tn, genX])
    dat = dat.drop(dat.columns[[0]], axis=1)
    dat['labels'] = np.concatenate([tn_l, genX_l])

    # Clean data set of unwanted characters
    dat.replace(to_replace=r'[0-9"\*><\',]',
                value=' ', regex=True, inplace=True)

    # Shuffle data set
    dat = dat.sample(frac=1).reset_index(drop=True)
    doc = dat.iloc[:, 0]

    train_txt, test_txt, train_label, test_label = train_test_split(
        doc, dat['labels'], test_size=.2, train_size=.8, shuffle=True)
    return train_txt, test_txt, train_label, test_label


'''
LSTM Model
'''

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 50
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# truncate and pad the input sequences so that they are all in the same length for modeling
X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

# Converting categorical labels to numbers
Y = pd.get_dummies(df['Product']).values
print('Shape of label tensor:', Y.shape)

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# get accuracy score
accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

# prints out a plot of the Loss and Accuracy over time
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show();

# plt.title('Accuracy')
# plt.plot(history.history['acc'], label='train')
# plt.plot(history.history['val_acc'], label='test')
# plt.legend()
# plt.show();
