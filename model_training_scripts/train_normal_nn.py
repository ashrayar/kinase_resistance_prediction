#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:14:02 2023

@author: ashraya
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle as pkl
import matplotlib.pyplot as plt
#base_dir="/wynton/home/fraserlab/aravikumar/dms/"
base_dir="/Users/ashraya/dms/"
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len



infile=open(base_dir+"input_ready_data_v1.pickle","rb")
infile=open(base_dir+"esm_mean_embed_morgan.pickle","rb")

data_df=pkl.load(infile)
data_df.rename({'inhibitor_code':'fitness','fitness':'inhibitor_code'},inplace=True,axis='columns')
infile.close()
X=[]
Y=[]
for index,row in data_df.iterrows():
    row_to_add=[]
    for i in row['sequence_code']:
        row_to_add.append(i)
    for i in row['inhibitor_code']:
        row_to_add.append(int(i))
    X.append(row_to_add)
    Y.append(row['fitness'])

X=np.array(X)
Y=np.array(Y)
Y=Y.reshape(-1,1)
Y = StandardScaler().fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)


nn_model=nn.Sequential(
      nn.Linear(7051, 3500),
      nn.ReLU(),
      nn.Linear(3500, 1000),
      nn.ReLU(),
      nn.Linear(1000, 250),
      nn.ReLU(),
      nn.Linear(250, 1)
      )
print(nn_model)
batch_size = 32
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.002)
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)




n_epoch = 20
losses=[]
plot_x=[]
plot_y=[]
for i in range(n_epoch):
    for j, data in enumerate(train_dataloader, 0):
        inputs, targets = data
        targets = targets.reshape((targets.shape[0], 1))
        y_pred = nn_model(inputs)
        step_loss = mse_loss(y_pred, targets)
        #step_loss = l1_loss(y_pred,targets)
        if i==n_epoch-1:
            plot_x.append(y_pred)
            plot_y.append(targets)
        losses.append(step_loss.item())
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

pklfile=open(base_dir+"lr_0.002_epoch_20_losses.pickle","wb")
pkl.dump(losses,pklfile)
pklfile.close()

infile=open(base_dir+"esm_morgan_lr_0.002_epoch_20_losses.pickle","rb")

losses=pkl.load(infile)
plt.plot(losses[:])
# plt.ylabel('loss')
# plt.xlabel('training step')
