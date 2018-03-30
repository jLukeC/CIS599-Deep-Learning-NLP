import pandas as pd
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


train = pd.read_csv('data/train.csv')


#
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

#def tokenize(text): return text.translate(None, string.punctuation).split()

tfidf_vectorizer = TfidfVectorizer()
train_tfidfs = tfidf_vectorizer.fit_transform(train["comment_text"])

X = train_tfidfs
y = train['toxic']

mnb = MultinomialNB()
y_pred = mnb.fit(X,y).predict(X)

print(((y==y_pred).sum() / len(y_pred)))
