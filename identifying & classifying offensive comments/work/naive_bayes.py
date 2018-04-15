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
test = pd.read_csv('data/train.csv')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


vectorizer = CountVectorizer()
train_vectorized = vectorizer.fit_transform(train["comment_text"])
test_vectorized = vectorizer.transform(test["comment_text"])



train_submission = train.copy()
test_submission = test.copy()


def train_naive_bayes(X_train,y_trains,X_test,y_tests):

    for label in labels:
        y_train = y_trains[label]
        y_test = y_tests[label]
        
        mnb = MultinomialNB()
        mnb.fit(X_train,y_train)
        y_pred_train = mnb.predict(X_train)
        y_pred_test = mnb.predict(X_test)
        
        train_submission[label] = y_pred_train
        test_submission[label] = y_pred_test
         
        print(label,roc_auc_score(y_train,y_pred_train))

train_naive_bayes(train_vectorized,train[labels],test_vectorized,test[labels])

print("Training Total ROC AUC Avg:",roc_auc_score(train_submission[labels],train[labels]))

train_submission.to_csv("submissions/train_naive_bayes.csv")
test_submission.to_csv("submissions/test_naive_bayes.csv")
