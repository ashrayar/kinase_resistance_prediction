#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:13:06 2023

@author: ashraya
"""

import pickle as pkl
import xgboost as xgb
from scipy.stats import pearsonr
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import numpy as np


base_dir = "/Users/ashraya/dms/"
# base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
#ESM + distance data preparation
###############################################
infile = open(base_dir+"esm_embed_distances_with_fitness_X_train.pkl","rb")
X_train = pkl.load(infile)
infile.close()
infile = open(base_dir+"esm_embed_distances_with_fitness_y_train.pkl","rb")
y_train = pkl.load(infile)
infile.close()

infile = open(base_dir+"esm_embed_distances_with_fitness_X_test.pkl","rb")
X_test = pkl.load(infile)
infile.close()
infile = open(base_dir+"esm_embed_distances_with_fitness_y_test.pkl","rb")
y_test = pkl.load(infile)
infile.close()
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


# Scrambled morgan fingerprint input
###############################################
esm_scramble_df = pd.read_csv("/Users/ashraya/dms/esm_embed_scrambled_fingerprint_with_fitness.csv",header = 0)
keys = ["Crizo", "Gle"]
holdout_df = esm_scramble_df[esm_scramble_df['key'].isin(keys)]
combined_df = esm_scramble_df[~esm_scramble_df['key'].isin(keys)]

combined_df['pos'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'(\d+)\w', r'\1', x))
holdout_positions = np.random.choice(combined_df['pos'].unique(), size=int(len(combined_df['pos'].unique()) * 0.2), replace=False)
holdout_positions_df = combined_df[combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
combined_df = combined_df[~combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
holdout_mutations = np.random.choice(amino_acids, size=int(len(amino_acids) * 0.1), replace=False)
combined_df['mut'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'\d+(\w)', r'\1', x))
holdout_mutations_df = combined_df[combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
combined_df = combined_df[~combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
holdout_df = pd.concat([holdout_positions_df, holdout_mutations_df, holdout_df])

X_train = combined_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_scrambled_fingerprints_X_train.pkl","wb")
pkl.dump(X_train, opfile)
opfile.close()
y_train = combined_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_scrambled_fingerprints_y_train.pkl","wb")
pkl.dump(y_train, opfile)
opfile.close()

X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_scrambled_fingerprints_X_test.pkl","wb")
pkl.dump(X_test, opfile)
opfile.close()
y_test = holdout_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_scrambled_fingerprints_y_test.pkl","wb")
pkl.dump(y_test, opfile)
opfile.close()
###############################################

#Xgboost decision tree
###############################################
best_parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}
best_xgb_regressor = xgb.XGBRegressor(**best_parameters)
print("Training...")
best_xgb_regressor.fit(X_train, y_train)
print("Predicting...")
test_predictions = best_xgb_regressor.predict(X_test)
train_predictions = best_xgb_regressor.predict(X_train)
pearson_train = pearsonr(y_train,train_predictions)
pearson_test = pearsonr(y_test, test_predictions)
print(f"\nSpearman R: {spearman.statistic}, p-value: {spearman.pvalue}")

###############################################
import matplotlib.pyplot as plt

plt.scatter(y_test,test_predictions)
###############################################