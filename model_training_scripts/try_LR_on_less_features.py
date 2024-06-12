#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:27:55 2023

@author: ashraya
"""
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.base import clone 
import matplotlib.pyplot as plt
from copy import deepcopy

#Extract LLR features
###################################################
esm_df = pd.read_csv("met_esm1b_llr_scores_of_interest.csv")
esm_df = esm_df[esm_df.score != 0.0] # remove 0.0 scores
esm_df['pos'] = esm_df['pos'].apply(lambda x: x - 1058) # normalizing position to start of kinase domain
# normalize pos_mut
esm_df['pos_mut'] = esm_df.apply(lambda x: str(x['pos']) + x['mut'][-1], axis=1)

esm_df = esm_df.drop(columns=["mut"])
# esm_df.rename(columns={'./model_weights/esm2_t36_3B_UR50D.pt': 'score'}, inplace=True)
esm_df = esm_df[['pos_mut', 'score']]
print(esm_df.head()) # Verification

scores_df = pd.read_csv("ex14_scores_unfiltered.tsv", sep='\t')
scores_df['pos_mut'] = scores_df['position'].astype(str) + scores_df['mutation'].astype(str)
scores_df['pos'] = scores_df['position']
scores_df = scores_df[['key', 'pos_mut', 'mean', 'pos']]
combined_df = pd.merge(esm_df, scores_df, on='pos_mut')
combined_df = combined_df[combined_df['key'] != 'Tiv']
esm_only_df = combined_df.copy()
###################################################


#Extract ThermoMPNN features
###################################################

# Type I vs Type II ThermoMPNN output
type1 = pd.read_csv("2wgj_mpnn.csv")
type2 = pd.read_csv("4eev_mpnn.csv")

# Drop garbage columns
type1 = type1.drop(columns=['Unnamed: 0'])
type2 = type2.drop(columns=['Unnamed: 0'])

# Kinase domain is only between residues 1059-1345
type1 = type1[(type1['pos'] >= 1059) & (type1['pos'] <= 1345)]
type2 = type2[(type2['pos'] >= 1059) & (type2['pos'] <= 1345)]

# Remove self mutations
type1 = type1[type1['wtAA'] != type1['mutAA']]
type2 = type2[type2['wtAA'] != type2['mutAA']]
type1 = type1.reset_index(drop=True)
type2 = type2.reset_index(drop=True)

# Difference in ddG
difference_df = pd.merge(type1, type2, on=['Mutation'], suffixes=('_type1', '_type2'))
difference_df['dddG'] = difference_df['ddG (kcal/mol)_type1'] - difference_df['ddG (kcal/mol)_type2']
difference_df['pos_mut'] = difference_df.apply(lambda x: str(x['pos_type1'] - 1058) + x['mutAA_type1'], axis=1)
difference_df = difference_df[['pos_mut', 'dddG']]

###################################################


#Get LLR+dddG data
###################################################
combined_df = pd.merge(esm_only_df, difference_df, on='pos_mut')
combined_df = combined_df.dropna()
esm_ddg_df = combined_df.copy()
###################################################


#Get LLR+interaction features data
###################################################
interaction_features_df = pd.read_csv("interaction_features.csv")
interaction_features_df = interaction_features_df.rename(columns={'Unnamed: 0': 'key'})
combined_df = pd.merge(esm_only_df, interaction_features_df, on=['key', 'pos'], how='left')
# For DMSO, all interaction features are 0s, so set NAs to 0
combined_df = combined_df.fillna(0)
esm_interaction_df = combined_df.copy()
# print(combined_df.head()) # Verification
###################################################

#Get LLR+interaction+dddG features data
###################################################
combined_df = pd.merge(esm_only_df, difference_df, on='pos_mut')
combined_df = combined_df.dropna()
combined_df = pd.merge(combined_df, interaction_features_df, on=['key', 'pos'], how='left')
# For DMSO, all interaction features are 0s, so set NAs to 0
combined_df = combined_df.fillna(0)
esm_interaction_ddg_df = combined_df.copy()
# print(combined_df.head()) # Verification
###################################################


mse_scores = []
spearman_scores = []
pearson_scores = []

lin_model = LinearRegression()

for i in range(5):

    # Hold outs
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
    X_train = train_df.drop(['key', 'pos_mut', 'mean'], axis=1)
    y_train = train_df['mean']
    X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
    y_test = holdout_df['mean']

    print("Shapes: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # Train
    lin_model.fit(X_train, y_train)
    
    y_pred = lin_model.predict(X_test)
    y_train_pred = lin_model.predict(X_train)
    
    mse_scores.append(mean_squared_error(y_test, y_pred))
    spearman_scores.append(stats.spearmanr(y_test, y_pred)[0])
    pearson_scores.append(stats.pearsonr(y_test, y_pred)[0])
    print("MSE: ", str(mse_scores[-1]))
    print("Spearman correlation: " + str(spearman_scores[-1]))
    print("Train Pearson correlation: " + str(stats.pearsonr(y_train, y_train_pred)[0]))
    print("Pearson correlation: " + str(pearson_scores[-1]))

    if pearson_scores[-1] == max(pearson_scores):
        best_model = deepcopy(lin_model)

print("Avg MSE: " + str(np.mean(mse_scores)))
print("Avg Spearman correlation: " + str(np.mean(spearman_scores)))
print("Avg Pearson correlation: " + str(np.mean(pearson_scores)))

inhibitors = ["Crizo", "Tepo", "A458", "NVP", "Mere", "Savo", "Cabo", "Camp", "Gle", "Glu", "DMSO"] # "Tiv",

esm_only_correlations = []

for inhibitor in inhibitors:
    data = combined_df[combined_df['key'] == inhibitor]
    test = data['mean']
    pred = best_model.predict(data.drop(['key', 'pos_mut', 'mean', 'pos'], axis=1))
    correlation = stats.pearsonr(test, pred)[0]
    esm_only_correlations.append(correlation)

plt.figure(figsize=(10, 6))
plt.bar(inhibitors, esm_only_correlations)
plt.xlabel('Inhibitors')
plt.ylabel('Correlations')
plt.title('Correlations of the Model with Each Inhibitor (ESM1b LLR Only)')
plt.show()