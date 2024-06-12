#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:59:01 2023

@author: ashraya
"""

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle as pkl
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
# base_dir = "/Users/ashraya/dms/"

embed_fing_df = pd.read_csv(base_dir+"esm_embed_distances_with_fitness.csv",header = 0)
# temp_cols=embed_fing_df.columns.tolist()
# new_cols=temp_cols[0:1281] + temp_cols[1283:] + [temp_cols[1282]] + [temp_cols[1281]]
# embed_fing_df=embed_fing_df[new_cols]
X = embed_fing_df.iloc[:,1:-2]
Y = embed_fing_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
#For scrambling
test_cols = X_test.columns.tolist()
for col in test_cols:
    if "_x" in col: #or col == '0':
        np.random.seed(24+X_test.columns.get_loc(col))
        X_test[col] = np.random.permutation(X_test[col].values)



# infile = open(base_dir+"esm_embed_fingerprints_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_fingerprints_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()

# #For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_x" in col: #or col == '0':
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)

# model_name = "linear_regressor_esm_distance_correct_split_no_scramble"
# model_name = "linear_regressor_esm_distance_correct_split_with_scramble"
model_name = "linear_regressor_esm_distance_random_split_no_scramble"
# model_name = "linear_regressor_esm_distance_random_split_with_scramble"
# model_name = "linear_regressor_esm_fingerprint_correct_split_no_scramble"
# model_name = "linear_regressor_esm_fingerprint_correct_split_with_scramble"
# model_name = "linear_regressor_esm_fingerprint_random_split_no_scramble"
# model_name = "linear_regressor_esm_fingerprint_random_split_with_scramble"
# model_name = "linear_regressor_esm_fingerprint_random_split_with_esm_scramble"

lin_model = pkl.load(open(base_dir+model_name+"_model.pkl","rb"))

y_test_predict = lin_model.predict(X_test)
pearson_test = pearsonr(y_test, y_test_predict)
print(model_name+" "+str(pearson_test))

# model_name = "decision_tree_esm_distance_correct_split_no_scramble"
# model_name = "decision_tree_esm_distance_correct_split_with_scramble"
model_name = "decision_tree_esm_distance_random_split_no_scramble"
# model_name = "decision_tree_esm_distance_random_split_with_scramble"
# model_name = "decision_tree_esm_fingerprint_correct_split_no_scramble"
# model_name = "decision_tree_esm_fingerprint_correct_split_with_scramble"
# model_name = "decision_tree_esm_fingerprint_random_split_no_scramble"
# model_name = "decision_tree_esm_fingerprint_random_split_with_scramble"
# model_name = "decision_tree_esm_fingerprint_random_split_with_esm_scramble"
best_xgb_regressor=pkl.load(open(base_dir+model_name+"_model.pkl","rb"))
test_predictions = best_xgb_regressor.predict(X_test)
pearson_test = pearsonr(y_test, test_predictions)
print(model_name+" "+str(pearson_test))

# model_name = "neural_network_esm_distance_correct_split_no_scramble"
# model_name = "neural_network_esm_distance_correct_split_with_scramble"
model_name = "neural_network_esm_distance_random_split_no_scramble"
# model_name = "neural_network_esm_distance_random_split_with_scramble"
# model_name = "neural_network_esm_fingerprint_correct_split_no_scramble"
# model_name = "neural_network_esm_fingerprint_correct_split_with_scramble"
# model_name = "neural_network_esm_fingerprint_random_split_no_scramble"
# model_name = "neural_network_esm_fingerprint_random_split_with_scramble"
# model_name = "neural_network_esm_fingerprint_random_split_with_esm_scramble"
best_model = pkl.load(open(base_dir+model_name+"_model.pkl","rb"))
test_predict = best_model.predict(X_test)
pearson_test = pearsonr(test_predict,y_test)
print(model_name+" "+str(pearson_test))
