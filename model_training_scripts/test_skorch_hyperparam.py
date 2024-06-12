#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:30:56 2023

@author: ashraya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
from skorch import NeuralNetRegressor
from skorch.helper import predefined_split

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units = [3000,1500,500,100]):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], hidden_units[3])
        self.fc5 = nn.Linear(hidden_units[3], output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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
    
    
param_grid = {
    'optimizer__lr': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'module__hidden_units': [(3000,1500,500,100), (4096,2048,512,128), (2048,1024,512,128)],
    'max_epochs': [10, 20, 30, 40, 50],
}

# base_dir="/Users/ashraya/dms/"
base_dir="/home/aravikumar/dms/"

# infile=open(base_dir+"esm_mean_embed_morgan.pickle","rb")

# data_df=pkl.load(infile)
# data_df.rename({'inhibitor_code':'fitness','fitness':'inhibitor_code'},inplace=True,axis='columns')
# infile.close()
# X=[]
# Y=[]
# for index,row in data_df.iterrows():
#     row_to_add=[]
#     for i in row['sequence_code']:
#         row_to_add.append(i)
#     for i in row['inhibitor_code']:
#         row_to_add.append(int(i))
#     X.append(row_to_add)
#     Y.append(row['fitness'])

# X=np.array(X)
# Y=np.array(Y)
# Y=Y.reshape(-1,1)
# Y = StandardScaler().fit_transform(Y)
# np.save("features_X.npy",X)
# np.save("targets_Y.npy",Y)


X = np.load(base_dir+"features_X.npy")
Y = np.load(base_dir+"targets_Y.npy")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)

input_size = X_train.shape[1]
output_size = 1
X_train=X_train.astype(float)
y_train=y_train.astype(float)
print("X_train shape = "+str(X_train.shape))
print("y_train shape = "+str(y_train.shape))
print("Data has been read")
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
# train_dataset = Data(X_train,y_train)


# nn_model=nn.Sequential(
#       nn.Linear(2304, 3000),
#       nn.ReLU(),
#       nn.Linear(3000, 500),
#       nn.ReLU(),
#       nn.Linear(500, 100),
#       nn.ReLU(),
#       nn.Linear(100, 1)
#       )
model = NeuralNetRegressor(module=CustomModel, module__input_size=input_size, module__output_size=output_size, criterion = nn.MSELoss(), optimizer = torch.optim.SGD)
# model = NeuralNetRegressor(nn_model, criterion = nn.MSELoss(), optimizer = torch.optim.SGD, max_epochs=20, batch_size=32)

# y_train_new = np.array([y for X, y in iter(train_dataset)])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=64, cv=3 , scoring = "neg_mean_squared_error")
grid_result = grid.fit(X=X_train_tensor, y=y_train_tensor)
opfile=open(base_dir+"grid_search_results.csv","w")
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
opfile.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
opfile.write("\n")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    opfile.write("%f (%f) with: %r" % (mean, stdev, param))
    opfile.write("\n")
opfile.close()