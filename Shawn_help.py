'''
--   OSU    --
--SHAWN HELP--
Shengxuan Wang
wangshe@oregonstate.edu
'''
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# A. Tools
# split
def split_xy(df, NameOfY):
    real_y = df[NameOfY]
    x = df.drop([NameOfY], axis=1)
    return (x, real_y)

# one hot encoder
def onehot_help(x, str_list):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(x[str_list].values)

    # the result data
    result = enc.transform(x[str_list].values).toarray()
    # the original label of them
    # labels = np.array(enc.categories_).ravel()

    col_names = []
    for col in str_list:
        for val in x[col].unique():
            col_names.append("{}_{}".format(col, val))

    return pd.DataFrame(data = result, columns=col_names, dtype=int)

def onehot(x, str_list, non_onehot_list):
    x_oh = onehot_help(x, str_list)
    df = pd.concat([x_oh, x[non_onehot_list]], axis=1)
    return df

# round score
def Rscore(clf, x, y):
    original_score = clf.score(x, y)
    rscore = format(original_score, '.4f')
    return rscore

# B. Classifiers
RS = 22     # random state

# 1 Dummy majority
def dummyM(x, y):
    clf = DummyClassifier(strategy="most_frequent", random_state=RS)
    clf.fit(x, y)
    return clf

# 2 Dummy distribution
def dummyD(x, y):
    clf = DummyClassifier(strategy="stratified", random_state=RS)
    clf.fit(x, y)
    return clf

# 3 RandomForestClassifier
def forest(x, y, n=50, classW = None):
    clf = RandomForestClassifier(n_estimators=n, random_state=RS, class_weight=classW)
    clf = clf.fit(x, y)
    return clf

# 4 KNN
def KNN(x, y):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf = clf.fit(x, y)
    return clf

# Scale the data
def scale(x):
    cols = x.columns
    scaler = StandardScaler()  
    scaler.fit(x) 
    x = scaler.transform(x)
    # avoid warning, transform it back to df
    x = pd.DataFrame(x, columns=cols)
    return x

# 5 Multi-layer Perceptron Classifier (with scale the data)
def MLPClf(x, y):
    x = scale(x)
    clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=400, hidden_layer_sizes=(5,), random_state=RS)
    clf = clf.fit(x, y)
    return clf

# 6 Decision Trees
def DTree(x, y):
    clf = DecisionTreeClassifier(random_state=RS)
    clf.fit(x,y)
    return clf

# 7 Logistic Regression
def LR(x, y, p, s, classW = None):
    clf = LogisticRegression(random_state=RS, penalty=p, solver=s, class_weight=classW)
    clf.fit(x, y)
    return clf