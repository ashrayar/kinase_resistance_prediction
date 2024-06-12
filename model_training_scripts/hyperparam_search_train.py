#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:25:29 2023

@author: ashraya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
from skorch import NeuralNetClassifier

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
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

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, test_loader, num_epochs, learning_rate, batch_size, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
            labels = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Print training loss or other relevant information for monitoring

    # Evaluate the model on the test set and return an evaluation metric
    predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        # Calculate your evaluation metric (e.g., accuracy, AUC, etc.) for the predictions
        # You can use scikit-learn or other libraries to calculate the metric
    correlation_coefficient, _ = pearsonr(predictions, true_labels)
    return correlation_coefficient

# base_dir="/home/aravikumar/dms/"
base_dir="/Users/ashraya/dms/"

param_dist = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'hidden_units': [(3000,1500,500,100), (4096,2048,512,128), (2048,1024,512,128)],
    'num_epochs': [10, 20, 30, 40, 50],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

print("Data has been read")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=26)


input_size = X_train.shape[1]  # Adjust input size according to your dataset
output_size = 1  # For binary classification

print("Instantiating model")
# Instantiate and run RandomizedSearchCV
#model = CustomModel(input_size, hidden_units=(3000,1500,500,100), output_size=output_size)
model = NeuralNetClassifier(CustomModel, criterion = nn.MSELoss(), optimizer = torch.optim.SGD)
print("Peforming Randomized CV")
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy')
random_search.fit(X_train, y_train)

print("Randomized CV done")
# Retrieve the best hyperparameters
best_params = random_search.best_params_
best_score = random_search.best_score_
best_model = random_search.best_estimator_

# Train the best model with the best hyperparameters
best_num_epochs = best_params['num_epochs']
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
print('best epochs = '+str(best_num_epochs))
print('best learning rate = '+str(best_learning_rate))
# print('best model = '+str(best_params))
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=best_batch_size, shuffle=True)
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=best_batch_size, shuffle=True)
print("Model evaluation starting")
model.to(device)
best_evaluation_metric = train_and_evaluate_model(best_model, X_train, y_train, X_test, y_test, test_dataloader, best_num_epochs, best_learning_rate, best_batch_size, device)

print(f'Best hyperparameters: {best_params}')
print(f'Best evaluation metric: {best_evaluation_metric}')