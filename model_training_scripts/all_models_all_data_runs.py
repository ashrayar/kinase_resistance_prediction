#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:10:04 2023

@author: ashraya
"""

import pickle as pkl
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np

# base_dir = "/Users/ashraya/dms/"
base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
print("Loading data")
#ESM + distance data preparation
###############################################
# infile = open(base_dir+"esm_embed_distances_with_fitness_X_train.pkl","rb")
# X_train = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_distances_with_fitness_y_train.pkl","rb")
# y_train = pkl.load(infile)
# infile.close()

# infile = open(base_dir+"esm_embed_distances_with_fitness_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_distances_with_fitness_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()

# #For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col:
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)

###############################################

#ESM data preparation
###############################################
# infile = open(base_dir+"esm_embed_with_fitness_X_train.pkl","rb")
# X_train = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_with_fitness_y_train.pkl","rb")
# y_train = pkl.load(infile)
# infile.close()

# infile = open(base_dir+"esm_embed_with_fitness_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_with_fitness_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()

#For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col:
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)

###############################################

#ESM difference data preparation
###############################################
# infile = open(base_dir+"esm_difference_embed_only_X_train.pkl","rb")
# X_train = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_difference_embed_only_y_train.pkl","rb")
# y_train = pkl.load(infile)
# infile.close()

# infile = open(base_dir+"esm_difference_embed_only_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_difference_embed_only_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()

# #For scrambling
# test_cols = X_test.columns.tolist()
# X_test_scramble = X_test.copy()
# for col in test_cols:
#     np.random.seed(24+X_test.columns.get_loc(col))
#     X_test_scramble[col] = np.random.permutation(X_test[col].values)

###############################################

#ESM difference random split
###############################################
embed_diff_df = pd.read_csv(base_dir + "esm_difference_embed_with_fitness.csv", header = 0)

X = embed_diff_df.iloc[:,1:-2]
Y = embed_diff_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
#For scrambling
test_cols = X_test.columns.tolist()
X_test_scramble = X_test.copy()
for col in test_cols:
    np.random.seed(24+X_test.columns.get_loc(col))
    X_test_scramble[col] = np.random.permutation(X_test[col].values)

###############################################

#ESM + fingerprint data preparation
###############################################
# infile = open(base_dir+"esm_embed_fingerprints_X_train.pkl","rb")
# X_train = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_fingerprints_y_train.pkl","rb")
# y_train = pkl.load(infile)
# infile.close()

# infile = open(base_dir+"esm_embed_fingerprints_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_fingerprints_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()

# #For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col or col == '0':
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)

###############################################

#ESM + fingerprint scrambled data correct split
###############################################
# embed_fing_df = pd.read_csv(base_dir+"esm_embed_fingerprint_with_fitness_scrambled.csv",header = 0)
# keys = ["Crizo", "Gle"]
# holdout_df = embed_fing_df[embed_fing_df['key'].isin(keys)]
# combined_df = embed_fing_df[~embed_fing_df['key'].isin(keys)]

# combined_df['pos'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'(\d+)\w', r'\1', x))
# holdout_positions = np.random.choice(combined_df['pos'].unique(), size=int(len(combined_df['pos'].unique()) * 0.2), replace=False)
# holdout_positions_df = combined_df[combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
# combined_df = combined_df[~combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)

# amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# holdout_mutations = np.random.choice(amino_acids, size=int(len(amino_acids) * 0.1), replace=False)
# combined_df['mut'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'\d+(\w)', r'\1', x))
# holdout_mutations_df = combined_df[combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
# combined_df = combined_df[~combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
# holdout_df = pd.concat([holdout_positions_df, holdout_mutations_df, holdout_df])

# infile = open(base_dir+"esm_embed_fingerprints_X_train.pkl","rb")
# X_train = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_fingerprints_y_train.pkl","rb")
# y_train = pkl.load(infile)
# infile.close()

# infile = open(base_dir+"esm_embed_fingerprints_X_test.pkl","rb")
# X_test = pkl.load(infile)
# infile.close()
# infile = open(base_dir+"esm_embed_fingerprints_y_test.pkl","rb")
# y_test = pkl.load(infile)
# infile.close()
# X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
# opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_scrambled_X_test.pkl","wb")
# pkl.dump(X_test, opfile)
# opfile.close()
# y_test = holdout_df['mean']
# opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_scrambled_y_test.pkl","wb")
# pkl.dump(y_test, opfile)
# opfile.close()

###############################################


#Random split ESM_distance
###############################################
# embed_dist_df = pd.read_csv(base_dir+"esm_embed_distances_with_fitness_local.csv",header = 0)
# # #For ESM + distance dataframe
# X = embed_dist_df.iloc[:,:-1]
# Y = embed_dist_df.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
# #For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col:
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)

###############################################

#Random split ESM_fingerprint
###############################################
# embed_fing_df = pd.read_csv(base_dir+"esm_embed_fingerprint_with_fitness.csv",header = 0)
# # temp_cols=embed_fing_df.columns.tolist()
# # new_cols=temp_cols[0:1281] + temp_cols[1283:] + [temp_cols[1282]] + [temp_cols[1281]]
# # embed_fing_df=embed_fing_df[new_cols]
# X = embed_fing_df.iloc[:,:-1]
# Y = embed_fing_df.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
# #For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col or col == '0':
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)
###############################################

#Random split ESM
###############################################
# embed_df = pd.read_csv(base_dir+"esm_embed_with_fitness.csv",header = 0)
# # temp_cols=embed_fing_df.columns.tolist()
# # new_cols=temp_cols[0:1281] + temp_cols[1283:] + [temp_cols[1282]] + [temp_cols[1281]]
# # embed_fing_df=embed_fing_df[new_cols]
# X = embed_df.iloc[:,1:-2]
# Y = embed_df.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
# # For scrambling
# test_cols = X_test.columns.tolist()
# for col in test_cols:
#     if "_y" in col or col == '0':
#         np.random.seed(24+X_test.columns.get_loc(col))
#         X_test[col] = np.random.permutation(X_test[col].values)
###############################################

filename = "esm_difference_random_split_results.csv"
opfile=open(base_dir+filename,"w")
opfile.write('model,Train pearson R,Test pearson R,scramble ESM test R,scramble inhibitor test R\n')

#Linear Regressor
###############################################
print("Linear Regression running")
from sklearn.linear_model import LinearRegression
# model_name = "linear_regressor_esm_distance_correct_split_no_scramble"
# model_name = "linear_regressor_esm_distance_correct_split_with_scramble"
# model_name = "linear_regressor_esm_distance_random_split_no_scramble"
# model_name = "linear_regressor_esm_distance_random_split_with_scramble"
# model_name = "linear_regressor_esm_fingerprint_correct_split_no_scramble"
# model_name = "linear_regressor_esm_fingerprint_correct_split_with_scramble"
# model_name = "linear_regressor_esm_fingerprint_random_split_no_scramble"
# model_name = "linear_regressor_esm_fingerprint_random_split_with_scramble"
# model_name = "linear_regressor_esm_random_split_no_scramble"
# model_name = "linear_regressor_esm_correct_split_no_scramble"
# model_name = "linear_regressor_esm_difference_correct_split"
model_name = "linear_regressor_esm_difference_random_split"
lin_model = LinearRegression().fit(X_train, y_train)
pkl.dump(lin_model, open(base_dir + model_name + '_model.pkl', 'wb'))

y_train_predict = lin_model.predict(X_train)
y_test_predict = lin_model.predict(X_test)
y_scramble_esm_predict = lin_model.predict(X_test_scramble)

pearson_train = pearsonr(y_train, y_train_predict)
pearson_test = pearsonr(y_test, y_test_predict)
pearson_scramble_test = pearsonr(y_test, y_scramble_esm_predict)

opfile.write(model_name + ","+ str(pearson_train[0]) + ","+ str(pearson_test[0]) + ","+ str(pearson_scramble_test[0])+",\n")

###############################################

#XGboost decision tree
###############################################
print("Decision tree running")
import xgboost as xgb
# model_name = "decision_tree_esm_distance_correct_split_no_scramble"
# model_name = "decision_tree_esm_distance_correct_split_with_scramble"
# model_name = "decision_tree_esm_distance_random_split_no_scramble"
# model_name = "decision_tree_esm_distance_random_split_with_scramble"
# model_name = "decision_tree_esm_fingerprint_correct_split_no_scramble"
# model_name = "decision_tree_esm_fingerprint_correct_split_with_scramble"
# model_name = "decision_tree_esm_fingerprint_random_split_no_scramble"
# model_name = "decision_tree_esm_fingerprint_random_split_with_scramble"
# model_name = "decision_tree_esm_random_split_no_scramble"
# model_name = "decision_tree_esm_difference_correct_split"
model_name = "decision_tree_esm_difference_random_split"
# model_name = "decision_tree_esm_correct_split_no_scramble"
best_parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
best_xgb_regressor = xgb.XGBRegressor(**best_parameters)
best_xgb_regressor.fit(X_train, y_train)
pkl.dump(best_xgb_regressor, open(base_dir + model_name + '_model.pkl', 'wb'))

test_predictions = best_xgb_regressor.predict(X_test)
train_predictions = best_xgb_regressor.predict(X_train)
y_scramble_esm_predict = best_xgb_regressor.predict(X_test_scramble)

pearson_train = pearsonr(y_train,train_predictions)
pearson_test = pearsonr(y_test, test_predictions)
pearson_scramble_test = pearsonr(y_test, y_scramble_esm_predict)

opfile.write(model_name + ","+ str(pearson_train[0]) + ","+ str(pearson_test[0]) + ","+ str(pearson_scramble_test[0])+",\n")

###############################################

#Neural Network
###############################################
print("Neural Network running")
from sklearn.neural_network import MLPRegressor
# model_name = "neural_network_esm_distance_correct_split_no_scramble"
# model_name = "neural_network_esm_distance_correct_split_with_scramble"
# model_name = "neural_network_esm_distance_random_split_no_scramble"
# model_name = "neural_network_esm_distance_random_split_with_scramble"
# model_name = "neural_network_esm_fingerprint_correct_split_no_scramble"
# model_name = "neural_network_esm_fingerprint_correct_split_with_scramble"
# model_name = "neural_network_esm_fingerprint_random_split_no_scramble"
# model_name = "neural_network_esm_fingerprint_random_split_with_scramble"
# model_name = "neural_network_esm_random_split_no_scramble"
# model_name = "neural_network_esm_difference_correct_split"
model_name = "neural_network_esm_difference_random_split"
# model_name = "neural_network_esm_correct_split_no_scramble"
#For fingeprint
# best_model = MLPRegressor(learning_rate_init=0.01, random_state = 108,hidden_layer_sizes=[16,4],solver = 'lbfgs',max_iter=10000)
#For distance
best_model = MLPRegressor(learning_rate_init=0.01, random_state = 108,hidden_layer_sizes=[8],solver = 'lbfgs',max_iter=10000)

best_model.fit(X_train,y_train)
pkl.dump(best_model, open(base_dir + model_name + '_model.pkl', 'wb'))

train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)
y_scramble_esm_predict = best_model.predict(X_test_scramble)

pearson_train = pearsonr(train_predict,y_train)
pearson_test = pearsonr(test_predict,y_test)

pearson_scramble_test = pearsonr(y_test, y_scramble_esm_predict)

opfile.write(model_name + ","+ str(pearson_train[0]) + ","+ str(pearson_test[0]) + ","+ str(pearson_scramble_test[0])+",\n")
opfile.close()
###############################################



