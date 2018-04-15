import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/train.csv')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


mdl = Pipeline([('vect', CountVectorizer()), ('ovr_mnb', OneVsRestClassifier(MultinomialNB()))])
mdl.fit(train["comment_text"],train[labels])
