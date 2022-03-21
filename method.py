'''
--OSU AI 539 Machine Learning Challenges
--Final Project
--Method - main file of the project
--Shengxuan Wang
--wangshe@oregonstate.edu
--3/14/2022
'''

import pandas as pd
from Shawn_help import split_xy, onehot
from Shawn_help import forest, LR
from baseline import super_onehot, super_onehot_for_A3, super_onehot_for_B1
from baseline import df, df_t, CLASS, minus_50, more_50

feature_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "relationship", "occupation", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
num_list = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
str_list = ["workclass", "education", "marital-status", "relationship", "occupation", "race", "sex", "native-country"]
non_onehot_list = num_list+[CLASS]

NUM_TREE = 40
PENALTY = 'l1'
SOLVER = 'liblinear'

df_missing = df.copy()
df_t_missing = df_t.copy()
df_whole_missing = pd.concat([df_missing, df_t_missing]).reset_index(drop=True)

# A missing value
# 1 drop all the example with missing value when training, always predict <=50K when there is missing value in validation set
print("****** README ******")
print("A B C means three challenges, 1,2,... means the solutions")
print("****** RESULT ******")
print("A.1")
df = df.dropna()
df_t = df_t.dropna()
df_whole = pd.concat([df, df_t]).reset_index(drop=True)

# print('after deleting missing value, the shape is')
# print(df_whole.shape)
# # (45222, 15)

df_whole_onehot = onehot(df_whole, str_list, non_onehot_list)
df_train = df_whole_onehot.sample(frac = 0.7, random_state=1)

df_rest = df_whole_onehot[~df_whole_onehot.index.isin(df_train.index)]

df_vali = df_rest.sample(frac = 0.5, random_state=1)
df_test = df_rest[~df_rest.index.isin(df_vali.index)]

# get the history shape to compute the proportion
# print('shape of train, vali, test')
# print(df_train.shape, df_vali.shape, df_test.shape)

# print('original shape in baseline are')
# print(df_train_base.shape, df_vali_base.shape, df_test_base.shape)

# print(df_vali_base.shape)
# print(df_vali_missing.shape)
# print(df_vali.shape)

# print('the shape of validation now is')
# print(df_vali.shape)
# # (6784, 105)
# # the original (before deleting missing value) shape of validation set is
# print('the shape of validation originally is')
# print(df_vali_base.shape)
# # (7326, 109)

# # now i need the distribution contains missing data in the original validation set, use it as the accuracy of the missing part
# df_vali_missing = df_whole_missing.sample(frac = 0.15, random_state=1)
# print(df_vali_missing.shape)
# print(df_vali_missing[CLASS].value_counts())
# # <=50K    5572
# # >50K     1754
# so the accuracy of it is 5572/(5572+1754)=0.76

def real_score(score):
    score = (score*6784+0.76*542)/7326
    return score

# split x and y
x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = real_score(score)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = real_score(score)
score = format(score, '.4f')
print("Logistic Regression: "+score)

# 2 Impute the missing values using the mode value of the feature.
print("\nA.2")

PRIVATE = df_missing.loc[2,'workclass']
PROF_SPE = df_missing.loc[4,'occupation']
USA = df_missing.loc[0,'native-country']

impute = {'workclass': PRIVATE, 'occupation': PROF_SPE, 'native-country': USA}
df_A2 = df_missing.fillna(value=impute)
df_t_A2 = df_t_missing.fillna(value=impute)

df_train, df_vali, df_test = super_onehot(df_A2, df_t_A2)

# split x and y
x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print("\nA.3")
df_A3 = df_missing.dropna(axis=1)
df_t_A3 = df_t_missing.dropna(axis=1)

df_train, df_vali, df_test = super_onehot_for_A3(df_A3, df_t_A3)

# split x and y
x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print("\nA.4")
# PRIVATE = df_missing.loc[2,'workclass']
# PROF_SPE = df_missing.loc[4,'occupation']

# impute = {'workclass': PRIVATE, 'occupation': PROF_SPE}
df_A4 = df_missing.copy()
df_t_A4 = df_t_missing.copy()

df_train, df_vali, df_test = super_onehot(df_A4, df_t_A4)

# split x and y
x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nB.1')
df_B1 = df_missing.drop(["capital-gain", "capital-loss"], axis=1)
df_t_B1 = df_t_missing.drop(["capital-gain", "capital-loss"], axis=1)

df_train, df_vali, df_test = super_onehot_for_B1(df_B1, df_t_B1)

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nB.2')
df_B2 = df_missing.copy()
df_t_B2 = df_t_missing.copy()

B2_list = ["capital-gain", "capital-loss"]
for i in B2_list:
    for n in range(df_B2.shape[0]):
        if df_B2.loc[n,i]!=0:
            df_B2.loc[n,i]=1
    for n in range(df_t_B2.shape[0]):
        if df_t_B2.loc[n,i]!=0:
            df_t_B2.loc[n,i]=1

df_train, df_vali, df_test = super_onehot(df_B2, df_t_B2)

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nB.3')

def real_score_B3(score):
    score = (score*6379+0.58*947)/7326
    return score

df_B3 = df_missing.copy()
df_t_B3 = df_t_missing.copy()

# get the history shape to compute the proportion
# print(df_B3.shape)
# print(df_t_B3.shape)
# # (32561, 15)
# # (16281, 15)

df_B3_nocap=df_B3[df_B3['capital-gain'].isin([0])]
df_B3_nocap=df_B3_nocap[df_B3_nocap['capital-loss'].isin([0])]

df_t_B3_nocap=df_t_B3[df_t_B3['capital-gain'].isin([0])]
df_t_B3_nocap=df_t_B3_nocap[df_t_B3_nocap['capital-loss'].isin([0])]

df_B3_cap = df_B3[~df_B3.index.isin(df_B3_nocap.index)]

df_t_B3_cap = df_t_B3[~df_t_B3.index.isin(df_t_B3_nocap.index)]

# print(df_B3_cap[CLASS].value_counts())
# # >50K     2450
# # <=50K    1781
# print(df_t_B3_cap[CLASS].value_counts())
# # >50K     1185
# # <=50K     901

# print((2450+1185)/(2450+1781+1185+901))

df_train, df_vali, df_test = super_onehot(df_B3_nocap, df_t_B3_nocap)

# print(df_vali.shape)
# (6379, 108)

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nC.1')
df_C1 = df_missing.copy()
df_t_C1 = df_t_missing.copy()

df_train, df_vali, df_test = super_onehot(df_C1, df_t_C1)
# split majority (<=50) and minority (>50)
df_train_maj = df_train[df_train[CLASS].isin([minus_50])]
df_train_mino = df_train[~df_train[CLASS].isin([minus_50])]
# undersample the majority
df_train_maj = df_train_maj.sample(frac = 0.5, random_state=1)
# combine them together
df_train = pd.concat([df_train_maj, df_train_mino])

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nC.2')
df_C2 = df_missing.copy()
df_t_C2 = df_t_missing.copy()

df_train, df_vali, df_test = super_onehot(df_C2, df_t_C2)
# split majority (<=50) and minority (>50)
df_train_maj = df_train[df_train[CLASS].isin([minus_50])]
df_train_mino = df_train[~df_train[CLASS].isin([minus_50])]

# augment more minority example
df_train_mino_copy = df_train_mino.copy()
df_train_mino_whole = pd.concat([df_train_mino, df_train_mino_copy])
# combine them together
df_train = pd.concat([df_train_maj, df_train_mino_whole])

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

score = clf_tree.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Logistic Regression: "+score)

print('\nC.3')
df_C3 = df_missing.copy()
df_t_C3 = df_t_missing.copy()

df_train, df_vali, df_test = super_onehot(df_C3, df_t_C3)

x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE, classW = {minus_50:2, more_50:1})
clf_LR = LR(x, y, p=PENALTY, s=SOLVER, classW = {minus_50:2, more_50:1})

score = clf_tree.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_v, y_v)
score = real_score_B3(score)
score = format(score, '.4f')
print("Logistic Regression: "+score)

# overall
print("****** OVERALL ******")
# PRIVATE = df_missing.loc[2,'workclass']
# PROF_SPE = df_missing.loc[4,'occupation']

# impute = {'workclass': PRIVATE, 'occupation': PROF_SPE}
df_A4 = df_missing.copy()
df_t_A4 = df_t_missing.copy()

df_train, df_vali, df_test = super_onehot(df_A4, df_t_A4)

# split x and y
x, y = split_xy(df_train, CLASS)
x_v, y_v = split_xy(df_vali, CLASS)
x_t, y_t = split_xy(df_test, CLASS)

clf_tree = forest(x, y, n=NUM_TREE)
clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

print("A.4 on Test set:")
score = clf_tree.score(x_t, y_t)
score = format(score, '.4f')
print("Random Forest: "+score)
score = clf_LR.score(x_t, y_t)
score = format(score, '.4f')
print("Logistic Regression: "+score)