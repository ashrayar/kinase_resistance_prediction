#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:05:17 2023

@author: ashraya
"""
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split


base_dir = "/Users/ashraya/dms/"
# base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
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
###############################################

#ESM + fingerprint data preparation
###############################################
infile = open(base_dir+"esm_embed_fingerprints_X_train.pkl","rb")
X_train = pkl.load(infile)
infile.close()
infile = open(base_dir+"esm_embed_fingerprints_y_train.pkl","rb")
y_train = pkl.load(infile)
infile.close()

infile = open(base_dir+"esm_embed_fingerprints_X_test.pkl","rb")
X_test = pkl.load(infile)
infile.close()
infile = open(base_dir+"esm_embed_fingerprints_y_test.pkl","rb")
y_test = pkl.load(infile)
infile.close()
###############################################

#Random split ESM_distance
###############################################
embed_dist_df = pd.read_csv("/Users/ashraya/dms/esm_embed_distances_with_fitness.csv",header = 0)
#For ESM + distance dataframe
X = embed_dist_df.iloc[:,1:-2]
Y = embed_dist_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)

###############################################

########################################################
# Train a MLP regressor
# Grid Search hyperparameters
learning_rate_init_values = [0.01, 0.001, 0.0001]
hidden_layer_sizes_values = [[4], [8], [16], [8,4], [16,4]]
random_state_vlaues = [101, 108]
#max_iter = [200, 1000, 10000]
parameter_space = {'learning_rate_init':learning_rate_init_values, 'hidden_layer_sizes':hidden_layer_sizes_values, 'random_state' :random_state_vlaues}
nn_model_parameterized = MLPRegressor(solver = 'lbfgs',max_iter=1000)
nn_model = GridSearchCV(nn_model_parameterized, parameter_space)
nn_model.fit(X_train, y_train)

# Best model parameters
nn_model.best_params_
opfile=open(base_dir+"sklearn_nn_esm_fingerprints_grid_search_results.csv","w")
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
# Make predictions
best_model = MLPRegressor(learning_rate_init=0.01, random_state = 108,hidden_layer_sizes=[16,4],solver = 'lbfgs',max_iter=10000)
best_model.fit(X_train,y_train)
train_predict = best_model.predict(X_train)
test_predict = best_model.predict(X_test)
pearson_train = pearsonr(train_predict,y_train)
pearson_test = pearsonr(test_predict,y_test)
########################################################