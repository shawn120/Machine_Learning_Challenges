'''
--OSU AI 539 Machine Learning Challenges
--Final Project
--Baseline
--Shengxuan Wang
--wangshe@oregonstate.edu
--3/14/2022
'''
import pandas as pd
from Shawn_help import split_xy, onehot, Rscore
from Shawn_help import forest, LR

CLASS = 'income'
feature_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "relationship", "occupation", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
num_list = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
str_list = ["workclass", "education", "marital-status", "relationship", "occupation", "race", "sex", "native-country"]
non_onehot_list = num_list+[CLASS]

df = pd.read_csv("adult_data.csv")
question_mark = df.iloc[27]["workclass"]
minus_50 = df.iloc[0,-1]
more_50 = df.iloc[32560,-1]
df = pd.read_csv("adult_data.csv", na_values=question_mark)
df_t = pd.read_csv("adult_test.csv", na_values=question_mark)

# do one hot together, then split to all the needed data then return
# in order to make the procesing easier (especially one hot encoding), concat training set and validation set together first,
# After processing, split them
def super_onehot(df, df_t):
    df_whole = pd.concat([df, df_t]).reset_index(drop=True)
    df_whole_onehot = onehot(df_whole, str_list, non_onehot_list)
    df_train = df_whole_onehot.sample(frac = 0.7, random_state=1)
    
    df_rest = df_whole_onehot[~df_whole_onehot.index.isin(df_train.index)]
    df_vali = df_rest.sample(frac = 0.5, random_state=1)
    df_test = df_rest[~df_rest.index.isin(df_vali.index)]

    return df_train, df_vali, df_test

def super_onehot_for_A3(df, df_t):
    str_list = ["education", "marital-status", "relationship", "race", "sex"]
    df_whole = pd.concat([df, df_t]).reset_index(drop=True)
    df_whole_onehot = onehot(df_whole, str_list, non_onehot_list)
    df_train = df_whole_onehot.sample(frac = 0.7, random_state=1)
    
    df_rest = df_whole_onehot[~df_whole_onehot.index.isin(df_train.index)]
    df_vali = df_rest.sample(frac = 0.5, random_state=1)
    df_test = df_rest[~df_rest.index.isin(df_vali.index)]

    return df_train, df_vali, df_test

def super_onehot_for_B1(df, df_t):
    num_list = ["age", "fnlwgt", "education-num", "hours-per-week"]
    non_onehot_list = num_list+[CLASS]
    df_whole = pd.concat([df, df_t]).reset_index(drop=True)
    df_whole_onehot = onehot(df_whole, str_list, non_onehot_list)
    df_train = df_whole_onehot.sample(frac = 0.7, random_state=1)
    
    df_rest = df_whole_onehot[~df_whole_onehot.index.isin(df_train.index)]
    df_vali = df_rest.sample(frac = 0.5, random_state=1)
    df_test = df_rest[~df_rest.index.isin(df_vali.index)]

    return df_train, df_vali, df_test

# print(df_vali.shape)

if __name__ == '__main__':

    # print(df.shape)
    # print(df_t.shape)
    df = df.dropna()
    df_t = df_t.dropna()
    df_train_base, df_vali_base, df_test_base = super_onehot(df, df_t)
    # print(df_train_base.shape)
    # print(df_vali_base.shape)
    # print(df_test_base.shape)
    # split x and y
    x, y = split_xy(df_train_base, CLASS)
    x_v, y_v = split_xy(df_vali_base, CLASS)
    x_t, y_t = split_xy(df_test_base, CLASS)

    # without any solution
    # first, examine all the hyperparameters
    # NUM_TREE = [10, 20, 30, 50]
    # PENALTY = ['l1', 'l2', 'none']
    # SOLVER = ['lbfgs', 'liblinear']
    NUM_TREE = 40

    PENALTY = 'l1'
    SOLVER = 'liblinear'

    # PENALTY = 'none'
    # SOLVER = 'lbfgs'

    # PENALTY = 'l2'
    # SOLVER = 'lbfgs'

    print('***** examine all the hyperparameters, then get the baseline *****') 
    clf_tree = forest(x, y, n=NUM_TREE)
    clf_LR = LR(x, y, p=PENALTY, s=SOLVER)

    def real_score(score):
        score = (score*6784+0*542)/7326
        return score

    print("on validation set:")
    score = clf_tree.score(x_v, y_v)
    score = real_score(score)
    score = format(score, '.4f')
    print("Random Forest: "+score)
    score = clf_LR.score(x_v, y_v)
    score = real_score(score)
    score = format(score, '.4f')
    print("Logistic Regression: "+score)
    print("on test set:")
    score = clf_tree.score(x_t, y_t)
    score = real_score(score)
    score = format(score, '.4f')
    print("Random Forest: "+score)
    score = clf_LR.score(x_t, y_t)
    score = real_score(score)
    score = format(score, '.4f')
    print("Logistic Regression: "+score)