#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:24:38 2023

@author: ashraya
"""


import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
# base_dir = "/Users/ashraya/dms/"


# combined_df = pd.read_csv(base_dir+"ESM_llr_ddg_atp_distance.csv", header = 0)
combined_df = pd.read_csv(base_dir+"ESM_llr_ddg_atp_resistance.csv", header = 0)
keys = ["Crizo", "Gle"]
holdout_df = combined_df[combined_df['key'].isin(keys)].copy()
train_df = combined_df[~combined_df['key'].isin(keys)].copy()

holdout_positions = np.random.choice(train_df['pos'].unique(), size=int(len(train_df['pos'].unique()) * 0.2), replace=False)
holdout_positions_df = train_df[train_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
train_df = train_df[~train_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
holdout_df = holdout_df.drop(['pos'], axis=1)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
holdout_mutations = np.random.choice(amino_acids, size=int(len(amino_acids) * 0.1), replace=False)
train_df['mut'] = train_df['pos_mut'].apply(lambda x: re.sub(r'\d+(\w)', r'\1', x))
holdout_mutations_df = train_df[train_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
train_df = train_df[~train_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)

# Verification
holdout_df = pd.concat([holdout_positions_df, holdout_mutations_df, holdout_df])
print('Total holdout:', len(holdout_df))
print('Total train:', len(train_df))
print('Split: (test fraction)', f"{len(holdout_df) / (len(holdout_df) + len(train_df)):.2f}")

# Train will be combined_df, test will be holdout_df
X_train = train_df.drop(['key', 'pos_mut', 'resistance'], axis=1)
y_train = train_df['resistance']
X_test = holdout_df.drop(['key', 'pos_mut', 'resistance'], axis=1)
y_test = holdout_df['resistance']


learning_rate_init_values = [0.01, 0.001, 0.0001]
hidden_layer_sizes_values = [[4], [8], [16], [8,4], [16,4]]
random_state_vlaues = [101, 108]
#max_iter = [200, 1000, 10000]
parameter_space = {'learning_rate_init':learning_rate_init_values, 'hidden_layer_sizes':hidden_layer_sizes_values, 'random_state' :random_state_vlaues}
# nn_model_parameterized = MLPRegressor(solver = 'lbfgs',max_iter=10000)
nn_model_parameterized = MLPClassifier(solver = 'lbfgs',max_iter=10000)
nn_model = GridSearchCV(nn_model_parameterized, parameter_space)
nn_model.fit(X_train, y_train)

opfile=open(base_dir+"sklearn_nn_llr_ddg_atp_dist_classify_grid_search_results.csv","w")
# summarize results
print("Best: %f using %s" % (nn_model.best_score_, nn_model.best_params_))
opfile.write("Best: %f using %s" % (nn_model.best_score_, nn_model.best_params_))
opfile.write("\n")
means = nn_model.cv_results_['mean_test_score']
stds = nn_model.cv_results_['std_test_score']
params = nn_model.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    opfile.write("%f (%f) with: %r" % (mean, stdev, param))
    opfile.write("\n")
opfile.close()

#Best: 0.017643 using {'hidden_layer_sizes': [16], 'learning_rate_init': 0.01, 'random_state': 108}
'''
best_model = MLPRegressor(learning_rate_init=0.01, random_state = 108,hidden_layer_sizes=[16],solver = 'lbfgs',max_iter=10000)
best_model.fit(X_train,y_train)
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)
pearson_train = pearsonr(train_predict,y_train)
pearson_test = pearsonr(test_predict,y_test)
'''
