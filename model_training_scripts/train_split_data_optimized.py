#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:06:54 2023

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

smiles = {'DMSO': 'CS(=O)C', 'Crizo': 'CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=C(N=CC(=C2)C3=CN(N=C3)C4CCNCC4)N', 'A458': 'CC1=C(C(=O)N(N1CC(C)(C)O)C2=CC=CC=C2)C(=O)NC3=NC=C(C=C3)OC4=C5C=CC(=CC5=NC=C4)OC', 'Cabo': 'COC1=CC2=C(C=CN=C2C=C1OC)OC3=CC=C(C=C3)NC(=O)C4(CC4)C(=O)NC5=CC=C(C=C5)F', 'Camp': 'CNC(=O)C1=C(C=C(C=C1)C2=NN3C(=CN=C3N=C2)CC4=CC5=C(C=C4)N=CC=C5)F', 'Gle': 'COCCNCC1=CN=C(C=C1)C2=CC3=NC=CC(=C3S2)OC4=C(C=C(C=C4)NC(=S)NC(=O)CC5=CC=C(C=C5)F)F', 'Glu': 'CN1C=C(C=N1)C2=CN3C(=NC=C3S(=O)(=O)N4C5=C(C=N4)N=CC(=C5)C6=CN(N=C6)C)C=C2', 'NVP': 'CN1C=C(C=N1)C2=NN3C(=NC=C3CC4=CC5=C(C=C4)N=CC=C5)C=C2', 'Tepo': 'CN1CCC(CC1)COC2=CN=C(N=C2)C3=CC=CC(=C3)CN4C(=O)C=CC(=N4)C5=CC=CC(=C5)C#N', 'Savo': 'CC(C1=CN2C=CN=C2C=C1)N3C4=NC(=CN=C4N=N3)C5=CN(N=C5)C', 'Tiv': 'C1CC2=C3C(=CC=C2)C(=CN3C1)C4C(C(=O)NC4=O)C5=CNC6=CC=CC=C65', 'Mere': 'CC1=CC=C(C(=O)N1C2=CC=C(C=C2)F)C(=O)NC3=CC(=C(C=C3)OC4=C(C=C5C(=C4)C=NN5C)C6=CNN=C6)F'}

# Convert SMILES to RDKit molecule objects
molecules = {inhib: Chem.MolFromSmiles(smile) for inhib, smile in smiles.items()}

# Calculate Morgan fingerprints with radius 2 (equivalent to ECFP4)
fingerprints = {inhib: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for inhib, mol in molecules.items()}

# Convert the fingerprints to bitstrings
bit_strings = {inhib: fp.ToBitString() for inhib, fp in fingerprints.items()}

# Convert to numpy arrays
bit_strings = {inhib: np.array(list(bit_string)).astype(int) for inhib, bit_string in bit_strings.items()}

# Convert to dataframe
mfp_df = pd.DataFrame(bit_strings).T
# print(mfp_df.head()) # Verification


# Load the datasets
embeddings_df = pd.read_csv('all_seq_esm_embeddings.csv', header=None)
embeddings_df = embeddings_df.rename(columns={0: 'pos_mut'})
embeddings_df['pos_mut'] = embeddings_df['pos_mut'].apply(lambda x: re.sub(r'(\d+)_\w_(\w)', r'\1\2', x))

scores_df = pd.read_csv('ex14_scores_filtered.tsv', sep='\t')
scores_df['pos_mut'] = scores_df['position'].astype(str) + scores_df['mutation'].astype(str)
scores_df = scores_df[['key', 'pos_mut', 'mean']]
scores_mfp_df = pd.merge(scores_df, mfp_df, left_on='key', right_index=True)
combined_df = pd.merge(embeddings_df, scores_mfp_df, on='pos_mut')

# Hold outs
keys = ["Crizo", "Gle"]
holdout_df = combined_df[combined_df['key'].isin(keys)]
combined_df = combined_df[~combined_df['key'].isin(keys)]

combined_df['pos'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'(\d+)\w', r'\1', x))
holdout_positions = np.random.choice(combined_df['pos'].unique(), size=int(len(combined_df['pos'].unique()) * 0.2), replace=False)
holdout_positions_df = combined_df[combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)
combined_df = combined_df[~combined_df['pos'].isin(holdout_positions)].drop(['pos'], axis=1)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
holdout_mutations = np.random.choice(amino_acids, size=int(len(amino_acids) * 0.2), replace=False)
combined_df['mut'] = combined_df['pos_mut'].apply(lambda x: re.sub(r'\d+(\w)', r'\1', x))
holdout_mutations_df = combined_df[combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)
combined_df = combined_df[~combined_df['mut'].isin(holdout_mutations)].drop(['mut'], axis=1)

# Verification
holdout_df = pd.concat([holdout_positions_df, holdout_mutations_df, holdout_df])
print('Total holdout:', len(holdout_df))
print('Total train:', len(combined_df))
print('Split: (test fraction)', f"{len(holdout_df) / (len(holdout_df) + len(combined_df)):.2f}")

# Train will be combined_df, test will be holdout_df
X_train = combined_df.drop(['key', 'pos_mut', 'mean'], axis=1)
y_train = combined_df['mean']
X_test = holdout_df.drop(['key', 'pos_mut', 'mean'], axis=1)
y_test = holdout_df['mean']

X_train_np=X_train.to_numpy()
y_train_np=y_train.to_numpy()
y_train_np=y_train_np.reshape(-1,1)
X_test_np=X_test.to_numpy()
y_test_np=y_test.to_numpy()
y_test_np=y_test_np.reshape(-1,1)
base_dir="/home/aravikumar/dms/"

input_size = X_train_np.shape[1]
output_size = 1
print("Data has been read")
X_train_tensor = torch.from_numpy(X_train_np.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train_np.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_np.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test_np.astype(np.float32))

#Best: -0.684058 using {'batch_size': 32, 'max_epochs': 30, 'module__hidden_units': (3000, 1500, 500, 100), 'optimizer__lr': 0.1}
nn_model = CustomModel(input_size, output_size, hidden_units= [3000, 1500, 500, 100])

print(nn_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn_model = nn_model.to(device=device)
batch_size = 32
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.1)
train_data = Data(X_train_np, y_train_np)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test_np, y_test_np)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

n_epoch = 30
losses=[]

for epoch in range(n_epoch):
    print("Epoch = "+str(epoch))
    for feat, target in train_dataloader:
        # inputs, targets = data
        # targets = targets.reshape((targets.shape[0], 1))
        feat = feat.to(device)
        target=target.to(device)
        y_pred = nn_model(feat)
        step_loss = mse_loss(y_pred, target)
        #step_loss = l1_loss(y_pred,targets)
        losses.append(step_loss.item())
        # print(step_loss.item())
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
torch.save(nn_model, base_dir+'correct_split_optimized_4_layer_crizo_gle.pt')
plt.figure()
plt.plot(losses[:])
plt.ylabel('loss')
plt.xlabel('training step')
plt.savefig(base_dir+"correct_split_optimized_losses_4_layer_crizo_gle.pdf",bbox_inches="tight")
plt.close()

print("Training complete... Now testing")
# nn_model_trained=torch.load("correct_split_optimized_5_layer_30.pt",map_location=torch.device('cpu'))
y_pred_list=[]
y_test_list=[]
with torch.no_grad():
    nn_model.eval()
    for feat, target in test_dataloader:
        feat = feat.to(device)
        y_test_pred = nn_model(feat)
        y_pred_list.append(y_test_pred.cpu().numpy())
        y_test_list.append(target.detach().cpu().numpy())

y_pred = list(itertools.chain(*y_pred_list))
y_pred_list=[]
for i in range(len(y_pred)):
    y_pred_list.append(y_pred[i][0])
y_test = list(itertools.chain(*y_test_list))
y_test_list=[]
for i in range(len(y_test)):
    y_test_list.append(y_test[i][0])
    
print(pearsonr(y_test_list,y_pred_list))