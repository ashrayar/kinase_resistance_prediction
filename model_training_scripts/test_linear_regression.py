#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:16:14 2023

@author: ashraya
"""

from sklearn.linear_model import LinearRegression
import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
import pickle as pkl
from scipy.stats import pearsonr

#ESM + distance data preparation
###############################################
embed_dist_df = pd.read_csv("/Users/ashraya/dms/esm_embed_distances_with_fitness.csv",header = 0)
keys = ["Crizo", "Gle"]
holdout_df = embed_dist_df[embed_dist_df['key'].isin(keys)]
combined_df = embed_dist_df[~embed_dist_df['key'].isin(keys)]

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
opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_X_train.pkl","wb")
pkl.dump(X_train, opfile)
opfile.close()
y_train = combined_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_y_train.pkl","wb")
pkl.dump(y_train, opfile)
opfile.close()

X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_X_test.pkl","wb")
pkl.dump(X_test, opfile)
opfile.close()
y_test = holdout_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_distances_with_fitness_y_test.pkl","wb")
pkl.dump(y_test, opfile)
opfile.close()

X_train_esm = X_train.iloc[:,0:1280]
X_test_esm = X_test.iloc[:,0:1280]
###############################################

#ESM + fingerprint data preparation
###############################################
embed_finger_df = pd.read_csv("/Users/ashraya/dms/esm_embed_fingerprint_with_fitness.csv",header = 0)
temp_cols=embed_finger_df.columns.tolist()
new_cols=temp_cols[0:1281] + temp_cols[1283:] + [temp_cols[1282]] + [temp_cols[1281]]
embed_finger_df=embed_finger_df[new_cols]

keys = ["Crizo", "Gle"]
holdout_df = embed_finger_df[embed_finger_df['key'].isin(keys)]
combined_df = embed_finger_df[~embed_finger_df['key'].isin(keys)]

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
opfile=open("/Users/ashraya/dms/esm_embed_fingerprints_X_train.pkl","wb")
pkl.dump(X_train, opfile)
opfile.close()
y_train = combined_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_fingerprints_y_train.pkl","wb")
pkl.dump(y_train, opfile)
opfile.close()

X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_fingerprints_X_test.pkl","wb")
pkl.dump(X_test, opfile)
opfile.close()
y_test = holdout_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_fingerprints_y_test.pkl","wb")
pkl.dump(y_test, opfile)
opfile.close()


# Hold out mutations and positions only
keys = []
holdout_df = embed_finger_df[embed_finger_df['key'].isin(keys)]
combined_df = embed_finger_df[~embed_finger_df['key'].isin(keys)]
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
opfile=open("/Users/ashraya/dms/esm_embed_fingerprint_no_inhib_holdout_X_train.pkl","wb")
pkl.dump(X_train, opfile)
opfile.close()
y_train = combined_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_fingerprint_no_inhib_holdout_y_train.pkl","wb")
pkl.dump(y_train, opfile)
opfile.close()

X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_fingerprint_no_inhib_holdout_X_test.pkl","wb")
pkl.dump(X_test, opfile)
opfile.close()
y_test = holdout_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_fingerprint_no_inhib_holdout_y_test.pkl","wb")
pkl.dump(y_test, opfile)
opfile.close()


###############################################

#Held out train and test data
###############################################

lin_model = LinearRegression().fit(X_train_esm, y_train)
lin_model_train_R2 = lin_model.score(X_train_esm, y_train)
lin_model_test_R2 = lin_model.score(X_test_esm, y_test)
y_train_predict = lin_model.predict(X_train_esm)
y_test_predict = lin_model.predict(X_test_esm)

pearsonr(y_train, y_train_predict)
pearsonr(y_test, y_test_predict)

from numpy.linalg import matrix_rank
matrix_rank(X_train.iloc[:,1280:].to_numpy())
matrix_rank(X_train.iloc[:10,:1280].to_numpy())
matrix_rank(embed_dist_df.iloc[:,1281:-2].to_numpy())
###############################################


# Random train test split
###############################################
from sklearn.model_selection import train_test_split
#For ESM + distance dataframe
X = embed_dist_df.iloc[:,1:-2]
Y = embed_dist_df.iloc[:,-1]

#For ESM + fingerprint dataframe
X = embed_finger_df.iloc[:,1:-2]
Y = embed_finger_df.iloc[:,-2]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)

###############################################


# hold out mutations and positions only
###############################################
embed_dist_df = pd.read_csv("/Users/ashraya/dms/esm_embed_distances_with_fitness.csv",header = 0)
keys = []
holdout_df = embed_dist_df[embed_dist_df['key'].isin(keys)]
combined_df = embed_dist_df[~embed_dist_df['key'].isin(keys)]
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
opfile=open("/Users/ashraya/dms/esm_embed_distances_no_inhib_holdout_X_train.pkl","wb")
pkl.dump(X_train, opfile)
opfile.close()
y_train = combined_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_distances_no_inhib_holdout_y_train.pkl","wb")
pkl.dump(y_train, opfile)
opfile.close()

X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
opfile=open("/Users/ashraya/dms/esm_embed_distances_no_inhib_holdout_X_test.pkl","wb")
pkl.dump(X_test, opfile)
opfile.close()
y_test = holdout_df['mean']
opfile=open("/Users/ashraya/dms/esm_embed_distances_no_inhib_holdout_y_test.pkl","wb")
pkl.dump(y_test, opfile)
opfile.close()


#Scrambled fingerprints
###############################################


###############################################

###############################################

#Linear Regression
###############################################

lin_model = LinearRegression().fit(X_train, y_train)
lin_model_train_R2 = lin_model.score(X_train, y_train)
lin_model_test_R2 = lin_model.score(X_test, y_test)
print(lin_model_train_R2)
print(lin_model_test_R2)
y_train_predict = lin_model.predict(X_train)
y_test_predict = lin_model.predict(X_test)

print(pearsonr(y_train, y_train_predict))
print(pearsonr(y_test, y_test_predict))

###############################################
