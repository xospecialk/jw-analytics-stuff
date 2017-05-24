import csv 
import pickle
import time

import numpy as np
from sklearn import datasets, svm
from sklearn.feature_extraction import DictVectorizer

publisher = list(csv.DictReader(open('ANALYTICS_ID_WATCHED_DURATION')))

vec = DictVectorizer()
vec.fit_transform(publisher).toarray()

print(vec.get_feature_names())



