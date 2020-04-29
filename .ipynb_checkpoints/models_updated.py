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
      

    

def mod_select_train(clfr_var, hypprm): #Train and tone model
    
    #Model Selection
    #Grid search
    gs = GridSearchCV(clfr_var, param_grid = hypprm, cv = 10, scoring = 'f1', refit = True) #Verbose shows u wats going on
    gs.fit(X_train, y_train)
        
    gs = gs.best_estimator_

    # K-fold cross validation
    cross_val_score(estimator = gs, X = X_train, y = y_train, cv = 10, scoring = 'f1')

    return gs

def models(X_train, y_train, X_test, y_test):
    
    preprocessor = vec()
    
    names = ['KNN', 'Naive Bayes', 'Multi layer Preceptrion']
    
    #classifiers = [LogisticRegression(max_iter=100), KNeighborsClassifier(), ComplementNB(), 
                   #MLPClassifier(hidden_layer_sizes = (20,3), max_iter= 100)]
    classifiers = [ (KNeighborsClassifier(),  {'classifier__n_neighbors': [2, 5, 10, 50, 100]}), (ComplementNB(),{'classifier__alpha': [0, 0.5, 1, 5]}), (MLPClassifier(hidden_layer_sizes = (20,3), max_iter= 100),{'classifier__activation': ['logistic', 'relu'], 
                 'classifier__solver': ['adam', 'sgd'], 
                 'classifier__alpha': [0, 0.001, 0.5, 1]})]
    
    
    out_file = open('Comp329ModelEvaluationReport.txt', 'w')
    
    out_file.writelines('Classifier Report' + '\n')
    
    for estimator, param in classifiers:
        print(estimator)
        pipe = Pipeline(steps = [('preprocessor', vec()),('classifier', estimator)])
        
        pipe.fit(X_train, y_train)   
        
        clf = GridSearchCV(pipe, param, cv=5, verbose=0, refit = True).fit(X_train, y_train)
        print(clf.best_estimator_)
        clf = clf.best_estimator_
        y_pred = clf.predict(X_test)
        
        out_file.writelines(' Results on test data' +'using '  + str(estimator) + 'tokenizer' + '\n')
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
