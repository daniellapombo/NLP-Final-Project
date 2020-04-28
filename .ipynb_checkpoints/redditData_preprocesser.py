# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:57:35 2020

@author: danie
"""

import pandas as pd
import numpy as np
import re

"""NLP Algorithms for Reddit Text Data

Objective: identify age demographics of the author and commentor
Goal: achieve a evalution metric of greater than 65%

Code:
    1. Import data
    2. Preprocess (clean) data 
        i. Of unwanted characters, and information
    3. Format data for NLP and ML algorithms
        i. Word2Vec, and Tf-idf
    4. Train various MLA and NLP's:
        i. KKN
        ii. Bayesian
        iii. CNN
        iv. NN
    5. Analyze efficency of algorithms on development set
        i. hyperparameter selection
        ii. model selection
    6. Process and analysize model on test data

"""

#Preprocessing
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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


def rdttxt_processing():
    tn, tn_l, genX, genX_l = import_txtF()
    
    #Create dataset (combining all data into one data set of text and labels)
    dat = pd.concat([tn, genX])
    dat = dat.drop(dat.columns[[0]], axis = 1)
    dat['labels'] = np.concatenate([tn_l, genX_l])
    
    #Clean data set of unwanted characters
    dat.replace(to_replace = r'[0-9"\*><\',]', value = ' ', regex = True, inplace = True)       
    
    #Shuffle data set
    dat = dat.sample(frac=1).reset_index(drop=True)
    
    doc = dat.iloc[:,0]
    doc = vec(doc) #Vectorize text data
    
    train_txt, test_txt, train_label, test_label = train_test_split(doc, dat['labels'], test_size= .2, train_size = .8, shuffle = True)
    
    return train_txt, test_txt, train_label, test_label

def vec(trn, tst= None):
    # Code inspired by https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn/36191362
    
    analyzer = TfidfVectorizer().build_analyzer()
    
    #Implements lemmalization funcationality to Vectorizer (tokenization & bag-of-words) method
    def stems(doc):
        ps = PorterStemmer() 
        return (ps.stem(w) for w in analyzer(doc))
    
    #Intialize vectorizer
    vectorize = TfidfVectorizer(stop_words = 'english', max_df = 1, min_df = 1, analyzer = stems)
    
    vec_trn = vectorize.fit_transform(trn) #Fits and Transforms training data
    vocab = vectorize.get_feature_names() #identifies the text (words) associated w/ each index value

    #return vec_trn, vec_tst
    return vec_trn


def main():
    rdttxt_processing()

   
main()

#do we thinkg '>', *' are important?
