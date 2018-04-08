import pandas as pd
import numpy as np
import re, string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


train = pd.read_csv('data/train.csv')
labels= ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


tfidf_vectorizer = CountVectorizer()
train_tfidfs = tfidf_vectorizer.fit_transform(train["comment_text"])


label_raw_accuracy_score = {}
label_roc_auc_score = {}

for label in labels:
    X = train_tfidfs
    y = train[label]
    mnb = MultinomialNB()
    y_pred = mnb.fit(X,y).predict(X)

    label_raw_accuracy_score[label] = accuracy_score(y,y_pred)
    label_roc_auc_score[label] = roc_auc_score(y,y_pred)
    
    print(label,label_raw_accuracy_score[label])
    print(label,label_roc_auc_score[label])

print(label_raw_accuracy_score)
print(label_roc_auc_score)