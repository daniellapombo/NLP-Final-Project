# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:34:30 2020

@author: danie
"""


import pandas as pd
import numpy as np
import re
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import skewnorm
from scipy import stats
import random

#Preprocessing
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#MLA and NLP Models
#from keras.models import Sequential
#from keras import layers
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#Word2Vec
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import KeyedVectors
from gensim.sklearn_api import W2VTransformer
from gensim.utils import lemmatize
 

#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB

#Model Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  auc

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


def import_txtF():
    #Import data sets and assign corresponding labels
        #'teen_processed.csv' - reddit teen text
        #'genx_processed.csv' - reddit GenX text
    
    teen = pd.read_csv('teen_processed.csv', header = [0])
    tn_label = np.zeros((teen.shape[0], 1), dtype=int)
    gX = pd.read_csv('genx_processed.csv', header = [0])
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
    
    train_txt, test_txt, train_label, test_label = train_test_split(doc, dat['labels'], test_size= .2, train_size = .8, shuffle = True)
   
    
    return train_txt, test_txt, train_label, test_label

def vec():
    # Code inspired by https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn/36191362
    
    analyzer = TfidfVectorizer().build_analyzer()
    
    #Implements lemmalization funcationality to Vectorizer (tokenization & bag-of-words) method
    def stems(doc):
        ps = PorterStemmer() 
        return (ps.stem(w) for w in analyzer(doc))
    
    #Intialize vectorizer
    return TfidfVectorizer(stop_words = 'english', max_df = 1, min_df = 1, analyzer = stems)
      

def wrd2vc():
    
    return W2VTransformer(size=300, window = 3, min_count = 3, sg = 1, trim_rule = lemmatize)
    
#
#class MeanEmbeddingVectorizer(object): #http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
#    def __init__(self, word2vec):
#        self.word2vec = word2vec
#        # if a text is empty we should return a vector of zeros
#        # with the same dimensionality as all the other vectors
#        self.dim = len(word2vec.itervalues().next())
#
#    def fit(self, X, y):
#        return self
#
#    def transform(self, X):
#        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#                    or [np.zeros(self.dim)], axis=0) for words in X])

def models(X_train, y_train, X_test, y_test):
    
    vectoriz = [vec(), wrd2vc()]
    
    names = ['LogisticReg', 'KNN', 'Naive Bayes', 'Multi layer Preceptrion']
    
    classifiers = [LogisticRegression(max_iter=100), KNeighborsClassifier(), ComplementNB(), 
                   MLPClassifier(hidden_layer_sizes = (20,3), max_iter= 100)]
    
    parameters = [{'classifier__penalty': ['l1', 'l2'],
                 'classifier__C': np.logspace(0, 4, 10)},
                {'classifier__n_neighbors': [2, 5, 10, 50, 100]}, 
                {'classifier__alpha': [0, 0.5, 1, 5]}, 
                {'classifier__activation': ['logistic', 'relu'], 
                 'classifier__solver': ['adam', 'sgd'], 
                 'classifier__alpha': [0, 0.001, 0.5, 1]}]
    
    parameters = [{'classifier__penalty': ['l2'],
                 'classifier__C': np.logspace(0, 4, 3)}]
    
#    #Record results/findings (data)
    out_file = open('Comp329ModelEvaluationReport.txt', 'w')
    
    out_file.writelines('Classifier Report' + '\n')
    
    for name, model, param in zip(names, classifiers, parameters):
        for preprocessor in vectoriz:
            print(classifiers)
            pipe = Pipeline(steps = [('preprocessor', preprocessor),('classifier', model)])
            pipe.fit(X_train, y_train)   
            
            clf = GridSearchCV(pipe, param, cv=5, verbose=0, refit = True).fit(X_train, y_train)
            print(clf.best_estimator_)
            clf = clf.best_estimator_
            y_pred = clf.predict(X_test)
            
            out_file.writelines(name + ' Results on test data' +'using '  + str(preprocessor) + 'tokenizer' + '\n')
            #out_file.writelines(clf + '\n')
            out_file.writelines('Accuracy ' + str(np.mean(y_pred == y_test)) + '\n')
            out_file.writelines(classification_report(y_test, y_pred) + '\n')
            PRE, REC, _ = precision_recall_curve(y_test, y_pred, pos_label = 1) #Precision recall curve returns precision, recall and its threshold
            AUC = auc(REC, PRE) #Compute area under precision recall curve
            out_file.writelines('AUC results ' + str(AUC) + '\n')
            out_file.writelines('\n')
    
    
    out_file.close()
    
    return None

def main():
    trn, test, trn_label, tst_label = rdttxt_processing()
    models(trn, trn_label, test, tst_label)
main()

#do we thinkg '>', *' are important?