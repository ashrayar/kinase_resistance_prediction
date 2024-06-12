#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:01:18 2023

@author: ashraya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:10:04 2023

@author: ashraya
"""


import pandas as pd
import re
import numpy as np
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.metrics import mean_squared_error


base_dir = "/Users/ashraya/dms/"
# base_dir = "/wynton/home/fraserlab/aravikumar/dms/"

eval_data = sys.argv[1]
print("Eval data = "+eval_data)

def get_correct_train_test_data(data_df):
    keys = ["Crizo", "Gle"]
    holdout_df = data_df[data_df['key'].isin(keys)].copy()
    train_df = data_df[~data_df['key'].isin(keys)].copy()

    holdout_positions = np.random.choice(train_df['pos'].unique(), size=int(len(train_df['pos'].unique()) * 0.2), replace=False)
    holdout_positions_df = train_df[train_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
    train_df = train_df[~train_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
    holdout_df = holdout_df.drop(['pos'], axis=1)

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    holdout_mutations = np.random.choice(amino_acids, size=int(len(amino_acids) * 0.1), replace=False)
    train_df['mut'] = train_df['pos_mut'].apply(lambda x: re.sub(r'\d+(\w)', r'\1', x))
    holdout_mutations_df = train_df[train_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
    train_df = train_df[~train_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)

    holdout_df = pd.concat([holdout_positions_df, holdout_mutations_df, holdout_df])
    

    X_train = train_df.drop(['key', 'pos_mut', 'mean'], axis=1)
    y_train = train_df['mean']
    X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
    y_test = holdout_df['mean']
    
    return X_train, y_train, X_test, y_test

def get_random_train_test_data(data_df):
    X = data_df.drop(['pos_mut','key','mean','pos'], axis = 1)
    print("X columns: ")
    print(X.columns)
    Y = data_df['mean']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)
    return X_train, y_train, X_test, y_test

def train_linear_regressor(lin_model, X_train, y_train, X_test, y_test):
    lin_model.fit(X_train, y_train)
    y_pred = lin_model.predict(X_test)
    y_train_pred = lin_model.predict(X_train)
    test_pearson_scores = stats.pearsonr(y_test, y_pred)[0]
    train_pearson_scores = stats.pearsonr(y_train, y_train_pred)[0]
    mse_score = mean_squared_error(y_test, y_pred)
    return train_pearson_scores, test_pearson_scores, mse_score

def train_neural_network(nn_model, X_train, y_train, X_test, y_test):
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    y_train_pred = nn_model.predict(X_train)
    test_pearson_scores = stats.pearsonr(y_test, y_pred)[0]
    train_pearson_scores = stats.pearsonr(y_train, y_train_pred)[0]
    return train_pearson_scores, test_pearson_scores

def train_xgboost(xgb_model, X_train, y_train, X_test, y_test):
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_train_pred = xgb_model.predict(X_train)
    test_pearson_scores = stats.pearsonr(y_test, y_pred)[0]
    train_pearson_scores = stats.pearsonr(y_train, y_train_pred)[0]
    return train_pearson_scores, test_pearson_scores
    
######################################################
if eval_data == "llr_only":
    input_df = pd.read_csv("ESM_llr_only_with_fitness.csv", header = 0)

elif eval_data == "llr_ddg":
    input_df = pd.read_csv("ESM_llr_ddg_with_fitness.csv", header = 0)

elif eval_data == "llr_prolif":
    input_df = pd.read_csv("ESM_llr_prolif_with_fitness.csv", header = 0)

elif eval_data == "llr_prolif_ddg":
    input_df = pd.read_csv("ESM_llr_ddg_prolif_with_fitness.csv", header = 0)

elif eval_data == "llr_atp_dist":
    input_df = pd.read_csv("ESM_llr_atp_distance_with_fitness.csv", header = 0)
    
elif eval_data == "llr_ddg_atp_dist":
    input_df = pd.read_csv("ESM_llr_ddg_atp_distance.csv", header = 0)

elif eval_data == "llr_vol_diff":
    input_df = pd.read_csv("ESM_llr_vol_diff_with_fitness.csv", header = 0)
    
elif eval_data == "llr_vol_diff_ddg_atp_dist":
    input_df = pd.read_csv("ESM_llr_ddg_atp_distance_vol_diff_with_fitness.csv", header = 0)

else:
    print("Wrong input")
    exit(1)
######################################################

#Initialize linear model
lin_model = LinearRegression()
# lin_model = LogisticRegression(random_state = 108, max_iter = 10000)
lin_model_pearson = []
lin_model_pearson_difference = [] 
#Initialize Neural Network

# nn_model = MLPRegressor(learning_rate_init=0.01, random_state = 108,hidden_layer_sizes=[16],solver = 'lbfgs',max_iter=10000)
# nn_model_pearson = []
# nn_model_pearson_difference = [] 
#Initialize XGBoost

# xgb_parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
# xgb_model = xgb.XGBRegressor(**xgb_parameters)
# xgb_model_pearson = []
# xgb_model_pearson_difference = [] 

for i in range(5):
    
    X_train, y_train, X_test, y_test = get_correct_train_test_data(input_df)
    # X_train, y_train, X_test, y_test = get_random_train_test_data(input_df)
    # Train linear model
    train_pearson, test_pearson, mse_score = train_linear_regressor(lin_model, X_train, y_train, X_test, y_test)
    lin_model_pearson.append(test_pearson)
    lin_model_pearson_difference.append(train_pearson - test_pearson)
    if lin_model_pearson[-1] == max(lin_model_pearson):
        best_lin_model = deepcopy(lin_model)
    print(test_pearson)
    
    #Train Neural Network
    # train_pearson, test_pearson = train_neural_network(nn_model, X_train, y_train, X_test, y_test)
    # nn_model_pearson.append(test_pearson)
    # nn_model_pearson_difference.append(train_pearson - test_pearson)
    # if nn_model_pearson[-1] == max(nn_model_pearson):
    #     best_nn_model = deepcopy(nn_model)
    
    #Train XGBoost
    # train_pearson, test_pearson = train_xgboost(xgb_model, X_train, y_train, X_test, y_test)
    # xgb_model_pearson.append(test_pearson)
    # xgb_model_pearson_difference.append(train_pearson - test_pearson)
    # if xgb_model_pearson[-1] == max(xgb_model_pearson):
    #     best_xgb_model = deepcopy(xgb_model)


avg_lr_pearson = np.mean(lin_model_pearson)
print("average pearson")
print(avg_lr_pearson)
# avg_nn_pearson = np.mean(nn_model_pearson)
# avg_xgb_pearson = np.mean(xgb_model_pearson)
avg_lr_pearson_diff = np.mean(lin_model_pearson_difference)
# avg_nn_pearson_diff = np.mean(nn_model_pearson_difference)
# avg_xgb_pearson_diff = np.mean(xgb_model_pearson_difference)

# inhibitors = ["Crizo", "Tepo", "A458", "NVP", "Mere", "Savo", "Cabo", "Camp", "Gle", "Glu", "DMSO"] # "Tiv",

# lr_inhib_correlations = []
# nn_inhib_correlations = []
# xgb_inhib_correlations = []

# for inhibitor in inhibitors:
#     data = input_df[input_df['key'] == inhibitor]
#     y_inhib_test = data['mean']
#     X_inhib_test = data.drop(['key', 'pos_mut', 'mean', 'pos'], axis=1)
    
#     lr_pred = best_lin_model.predict(X_inhib_test)
#     correlation = stats.pearsonr(y_inhib_test, lr_pred)[0]
#     lr_inhib_correlations.append(correlation)
    
#     nn_pred = best_nn_model.predict(X_inhib_test)
#     correlation = stats.pearsonr(y_inhib_test, nn_pred)[0]
#     nn_inhib_correlations.append(correlation)
    
#     xgb_pred = best_xgb_model.predict(X_inhib_test)
#     correlation = stats.pearsonr(y_inhib_test, xgb_pred)[0]
#     xgb_inhib_correlations.append(correlation)

# opfile1 = open(base_dir + 'new_features_models_inputs_tests_random_split.csv', 'a')
# opfile2 = open(base_dir + 'new_features_models_inputs_inhib_correlations_random_split.csv', 'a')
# opfile1.write('input_features,avg lr test,avg nn test,avg xgb test,avg lr difference,avg nn difference,avg xgb difference\n')
# opfile2.write('input_features,model,Crizo,Tepo,A458,NVP,Mere,Savo,Cabo,Camp,Gle,Glu,DMSO\n')

# opfile1.write(eval_data+","+str(avg_lr_pearson)+","+str(avg_nn_pearson)+","+str(avg_xgb_pearson)+","+str(avg_lr_pearson_diff)+","+str(avg_nn_pearson_diff)+","+str(avg_xgb_pearson_diff)+"\n")
# opfile2.write(eval_data+",linear regressor")
# for i in lr_inhib_correlations:
#     opfile2.write(","+str(i))
# opfile2.write("\n")

# opfile2.write(eval_data+",neural network")
# for i in nn_inhib_correlations:
#     opfile2.write(","+str(i))
# opfile2.write("\n")

# opfile2.write(eval_data+",xgboost")
# for i in xgb_inhib_correlations:
#     opfile2.write(","+str(i))
# opfile2.write("\n")

# opfile1.close()
# opfile2.close()



