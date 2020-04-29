# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:57:35 2020
@author: danie
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

def import_txtF():
    #Import data sets and assign corresponding labels
        #'teen_processed.csv' - reddit teen text
        #'genx_processed.csv' - reddit GenX text
    
    teen = pd.read_csv('./data/teen_processed.csv', header = [0])
    tn_label = np.zeros((teen.shape[0], 1), dtype=int)
    gX = pd.read_csv('./data/genx_processed.csv', header = [0])
    gX_label = np.ones((gX.shape[0], 1), dtype=int)

    return teen, tn_label, gX, gX_label


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
    # doc = vec(doc)
    train_txt, test_txt, train_label, test_label = train_test_split(
        doc, dat['labels'], test_size=.2, train_size=.8, shuffle=True)

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


def models(X_train, y_train, X_test, y_test):

    vectoriz = [vec(X_train,y_train)]

    names = ['LogisticReg', 'KNN', 'Naive Bayes', 'Multi layer Preceptrion']

    classifiers = [LogisticRegression(max_iter=100), KNeighborsClassifier(), ComplementNB(),
                   MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=100)]

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
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_train, y_train)

            clf = GridSearchCV(pipe, param, cv=5, verbose=0, refit=True).fit(X_train, y_train)
            print(clf.best_estimator_)
            clf = clf.best_estimator_
            y_pred = clf.predict(X_test)

            out_file.writelines(name + ' Results on test data' + 'using ' + str(preprocessor) + 'tokenizer' + '\n')
            #out_file.writelines(clf + '\n')
            out_file.writelines(
                'Accuracy ' + str(np.mean(y_pred == y_test)) + '\n')
            out_file.writelines(classification_report(y_test, y_pred) + '\n')
            # Precision recall curve returns precision, recall and its threshold
            PRE, REC, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
            AUC = auc(REC, PRE)  # Compute area under precision recall curve
            out_file.writelines('AUC results ' + str(AUC) + '\n')
            out_file.writelines('\n')

    out_file.close()

    return None

def main():
    train_txt, test_txt, train_label, test_label =  rdttxt_processing()
    models(train_txt, train_label, test_txt, test_label)
    

   
main()
