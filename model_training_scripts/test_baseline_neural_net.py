#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:39:21 2023

@author: ashraya
"""

import pandas as pd
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import itertools
from scipy.stats import pearsonr

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units = [3000,1500,500,100]):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], hidden_units[3])
        # self.fc5 = nn.Linear(hidden_units[3], hidden_units[4])
        self.fc5 = nn.Linear(hidden_units[3], output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return x

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    
embeddings_df = pd.read_csv('all_seq_esm_embeddings.csv', header=None)
embeddings_df = embeddings_df.rename(columns={0: 'pos_mut'})
embeddings_df['pos_mut'] = embeddings_df['pos_mut'].apply(lambda x: re.sub(r'(\d+)_\w_(\w)', r'\1\2', x))

scores_df = pd.read_csv('ex14_scores_filtered.tsv', sep='\t')
scores_df['pos_mut'] = scores_df['position'].astype(str) + scores_df['mutation'].astype(str)
scores_df = scores_df[['key', 'pos_mut', 'mean']]
combined_df = pd.merge(embeddings_df, scores_df, on='pos_mut')