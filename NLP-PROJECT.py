# NLP FINAL PROJECT
# ANGIE GEORGARAS, PAULINA ADAMNSKI, DANIELLA POMBO, LINETTE MALIAKAL, ALEX ROSE, JACK BROOKS
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

GenX = pd.read_csv('./data/original/GenX_sub_data.csv')
Teen = pd.read_csv('./data/original/teenagers_sub_data.csv')

# Removes bots, comments with links, and makes all comments lowercase


def clean_data(data):
    if (data['User'] != 'AutoModerator'):
        data['Comment'] = data['Comment'].lower()
        if('https://' not in data['Comment']):
            return data
    else:
            data = data.drop('User')


GenX = GenX.apply(clean_data, axis=1)
Teen = Teen.apply(clean_data, axis=1)

# drop deleted comments and unnecesssary columns
GenX = GenX.dropna()
Teen = Teen.dropna()

GenX = GenX.drop(['Unnamed: 0', 'SubredditofOrgin',
                 'Submission', 'User'], axis=1)
Teen = Teen.drop(['Unnamed: 0', 'SubredditofOrgin',
                 'Submission', 'User'], axis=1)

GenX.to_csv('./data/genx_processed.csv')
Teen.to_csv('./data/teen_processed.csv')


def import_txtF():
    # Import data sets and assign corresponding labels
        # 'teen_processed.csv' - reddit teen text
        # 'genx_processed.csv' - reddit GenX text

    teen = pd.read_csv('./data/teen_processed.csv', header=[0])
    tn_label = np.zeros((teen.shape[0], 1), dtype=int)
    gX = pd.read_csv('./data/genx_processed.csv', header=[0])
    gX_label = np.ones((gX.shape[0], 1), dtype=int)

    return teen, tn_label, gX, gX_label


  def rdttxt_processing():
      tn, tn_l, genX, genX_l = import_txtF()

      # Create dataset (combining all data into one data set of text and labels)
      dat = pd.concat([tn, genX])
      dat = dat.drop(dat.columns[[0]], axis = 1)
      dat['labels'] = np.concatenate([tn_l, genX_l])

      # Clean data set of unwanted characters
      dat.replace(to_replace = r'[0-9"\*><\',]', value = ' ', regex = True, inplace = True)

      # Shuffle data set
      dat = dat.sample(frac=1).reset_index(drop=True)

      doc = dat.iloc[:,0]
      doc = vec(doc) #Vectorize text data

      train_txt, test_txt, train_label, test_label = train_test_split(doc, dat['labels'], test_size= .2, train_size = .8, shuffle = True)

      return train_txt, test_txt, train_label, test_label


def vec(trn, tst= None):
            # Code inspired by https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn/36191362

            analyzer = TfidfVectorizer().build_analyzer()

            # Implements lemmalization funcationality to Vectorizer (tokenization & bag-of-words) method
            def stems(doc):
                ps = PorterStemmer()
                return (ps.stem(w) for w in analyzer(doc))

            # remove stop words (ANGIE)

            # Intialize vectorizer
            vectorize = TfidfVectorizer(stop_words = 'english', max_df = 1, min_df = 1, analyzer = stems)

            vec_trn = vectorize.fit_transform(trn) #Fits and Transforms training data
            vocab = vectorize.get_feature_names() #identifies the text (words) associated w/ each index value

            # return vec_trn, vec_tst
            return vec_trn


# Naive Bayes Classifier
def NB():
    nbs = ComplementNB()
    nbs.fit(train_txt, train_label)

    print("naive bayes")

# Logistic Regression Classifier
def LR():
    lr = LogisticRegression()
    lr.fit(train_txt, train_label)

    print("logistic regression")

# CNN Classifier

# LSTM Classifier (ANGIE)


# Model Training/Tuning
def model_training():

    print('')

# Model Evaluation (GRIDSEARCH = ANGIE)
def model_evaluation():

    print('')

def main():
    rdttxt_processing()
        # Wild out B

main()
