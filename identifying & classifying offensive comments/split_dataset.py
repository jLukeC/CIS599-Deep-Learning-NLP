import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data.csv')

train, test = train_test_split(data, test_size=0.2)

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
