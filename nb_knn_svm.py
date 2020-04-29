#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import multiprocessing
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import re
import spacy
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from nltk.tokenize import regexp_tokenize 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
#Word2Vec
import gensim.downloader as api
from gensim import utils
import gensim.models
from gensim.models import Word2Vec
from gensim.sklearn_api import W2VTransformer
from nltk.tokenize import regexp_tokenize
#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB




# In[2]:



GenX = pd.read_csv('./data/original/GenX_sub_data.csv')
Teen = pd.read_csv('./data/original/teenagers_sub_data.csv')

def clean_data(data):
    if (data['User'] != 'AutoModerator'):
        data['Comment'] = data['Comment'].lower()
        if('https://' not in data['Comment']):
            return data
    else:
        data = data.drop('User')


# In[3]:


def rdttxt_processing(GenX, Teen):
    GenX = GenX.apply(clean_data, axis=1)
    Teen = Teen.apply(clean_data, axis=1)

    # drop deleted comments and unnecesssary columns
    GenX = GenX.dropna()
    Teen = Teen.dropna()

    gX = GenX.drop(['Unnamed: 0', 'SubredditofOrgin', 'Submission', 'User'], axis=1)
    teen = Teen.drop(['Unnamed: 0', 'SubredditofOrgin','Submission', 'User'], axis=1)
    tn_l = np.zeros((teen.shape[0], 1), dtype=int)
    teen['labels'] = tn_l
    
    gX_l = np.ones((gX.shape[0], 1), dtype=int)
    gX['labels'] = gX_l
  
    #Create dataset (combining all data into one data set of text and labels)
    dat = pd.concat([teen, gX])
    dat = dat.sample(frac=1).reset_index(drop=True)
    
    #Clean data set of unwanted characters
    dat['Comment'] = dat['Comment'].apply(lambda x: x.replace(r'[0-9"\*><\',]', ''))  
    dat['Comment'] = dat['Comment'].apply(lambda x: re.sub(' +', ' ',x))
    
    
    #Shuffle data set
    dat = dat.sample(frac=1).reset_index(drop=True)
    
    doc = dat.iloc[:,0]
   
    train_txt, test_txt, train_label, test_label = train_test_split(doc, dat['labels'], test_size= .2, train_size = .8,  random_state=42)
    return train_txt, test_txt, train_label, test_label


# In[4]:


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# In[5]:



trn, test, trn_label, tst_label = rdttxt_processing(GenX, Teen)

trn = trn.apply(regexp_tokenize, pattern="[\w']+")

model = gensim.models.Word2Vec(trn, size=200, min_count=1, workers=4)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

etree_w2v = Pipeline([
("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
("extra trees", ExtraTreesClassifier(n_estimators=200))])


# In[6]:


KNN = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("KNN", KNeighborsClassifier())])
nb = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
svm = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])


# In[8]:


nb_s = round(np.mean(cross_val_score(nb, test, tst_label))*100,2)
KNN_s = round(np.mean(cross_val_score(KNN, test, tst_label))*100,2)
SVM_s = round(np.mean(cross_val_score(svm, test, tst_label))*100,2)
print("Cross Validation Score Results \n\n" + 
      f"Naive Bayes:{nb_s} %\n"+
      f"KNN:{KNN_s} %\n" +
      f"SVM:{SVM_s} %\n"
     )


# In[ ]:




