'''
--OSU AI 539 Machine Learning Challenges
--Final Project
--Profile
--Shengxuan Wang
--wangshe@oregonstate.edu
--3/14/2022
'''
# from cmath import nan
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer as si
# import matplotlib.pyplot as plt

# def split_xy(df, NameOfY="income"):
#     real_y = df[NameOfY]
#     x = df.drop([NameOfY], axis=1)
#     return (x, real_y)

# def biny_income(y):
#     minus_50=y.iloc[0]
#     for i in range(y.shape[0]):
#         if y.iloc[i] == minus_50:
#             y.iloc[i] = 0
#         else:
#             y.iloc[i] = 1

# df = pd.read_csv("adult_data.csv")
# df_v = pd.read_csv("adult_test.csv")

# question_mark = df.iloc[27]["workclass"]

# # df = pd.read_csv("adult_data.csv", na_values=question_mark)
# # df_v = pd.read_csv("adult_test.csv", na_values=question_mark)

# df = pd.concat([df,df_v]).reset_index(drop=True)

# head_list = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "relationship", "occupation", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
# num_list = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
# str_list = ["workclass", "education", "marital-status", "relationship", "occupation", "race", "sex", "native-country", "income"]

# # check missing values
# missing_list = []
# for i in head_list:
#     if any(df[i].isnull()):
#         missing_list.append(i)

# print("the features have missing values are", missing_list)
# minus_50=df.iloc[0,-1]
# df_1 = df[df["income"]!=minus_50]
# df_0 = df[df["income"]==minus_50]

# # histogram
# plt.figure(0, figsize=(15, 8))
# n = 1
# for i in num_list:
#     plt.subplot(2,3,n)
#     plt.hist([df_1[i],df_0[i]], bins=30, stacked = True, label=[">50K","<=50K"])
#     plt.legend()
#     plt.title("%s" %i)
#     n+=1
# plt.savefig("histogram_class.png")

# plt.figure(1, figsize=(30, 30), dpi=200)
# n = 1
# for i in str_list:
#     plt.subplot(3,3,n)
#     plt.hist([df_1[i],df_0[i]], bins=30, stacked = True, label=[">50K","<=50K"])
#     plt.xticks(rotation=300)
#     plt.legend()
#     plt.title("%s" %i)
#     n+=1
# plt.savefig("histogram_discrete_class.png")

# plt.figure(2, figsize=(20,20))
# plt.hist([df_1["workclass"],df_0["workclass"]], bins=30, stacked = True, label=[">50K","<=50K"])
# plt.xticks(rotation=300)
# plt.legend()
# plt.title("workclass")
# plt.savefig("histogram_workclass_class.png")

# plt.figure(3, figsize=(25,25))
# plt.hist([df_1["occupation"],df_0["occupation"]], bins=30, stacked = True, label=[">50K","<=50K"])
# plt.xticks(rotation=300)
# plt.legend()
# plt.title("occupation")
# plt.savefig("histogram_occupation_class.png")

# plt.figure(4, figsize=(25,25))
# plt.hist([df_1["native-country"],df_0["native-country"]], bins=30, stacked = True, label=[">50K","<=50K"])
# plt.xticks(rotation=300)
# plt.legend()
# plt.title("native-country")
# plt.savefig("histogram_native-country_class.png")

# plt.figure(5)
# plt.hist([df_1["capital-gain"],df_0["capital-gain"]], bins=50, stacked = True, label=[">50K","<=50K"])
# plt.legend()
# plt.title("capital-gain")
# plt.savefig("histogram_capital-gain_class.png")

# plt.figure(6)
# plt.hist([df_1["capital-loss"],df_0["capital-loss"]], bins=50, stacked = True, label=[">50K","<=50K"])
# plt.legend()
# plt.title("capital-loss")
# plt.savefig("histogram_capital-loss_class.png")

# # if delete missing values, comment this line if what the data be original 
# # df = df.dropna()

# # boxplot
# plt.figure(7, figsize=(20, 15))
# n = 1
# for i in num_list:
#     plt.subplot(2,3,n)
#     plt.boxplot(df[i])
#     plt.title("%s" %i)
#     n+=1
# plt.savefig("boxplot.png")

# # # what are they shapes?
# # print("what are they shapes?")
# # shape_train = df.shape
# # shape_valid = df_v.shape
# # print(shape_train)
# # print(shape_valid)

# # # take a look
# # print("take a look")
# # print(df.head())

# # # what are they type?
# # print("what are they type?")
# # for i in head_list:
# #     print("the type of %s" %i)
# #     print(type(df.iloc[0].loc[i]))



# # describe
# print("descripe")
# print(df.describe())

# # histogram
# plt.figure(8, figsize=(15, 8))
# n = 1
# for i in num_list:
#     plt.subplot(2,3,n)
#     plt.hist(df[i], bins=30)
#     plt.title("%s" %i)
#     n+=1
# plt.savefig("histogram.png")

# plt.figure(9, figsize=(30, 30), dpi=200)
# n = 1
# for i in str_list:
#     plt.subplot(3,3,n)
#     plt.hist(df[i], bins=30)
#     plt.xticks(rotation=300)
#     plt.title("%s" %i)
#     n+=1
# plt.savefig("histogram_discrete.png")

# for i in str_list:
#     print("the counts of %s is" %i)
#     print(df[i].value_counts())
#     print("*****")


# # df_x, df_y = split_xy(df)

# # # avoid copy warning
# # yy = df_y.copy()

# # biny_income(yy)

# # # # # print(df_train_y)
# # print(yy.value_counts())

# # print(37155/(37155+11687))
# # print(11687/(37155+11687))
