#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:57:43 2023

@author: ashraya
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle as pkl
import matplotlib.pyplot as plt
base_dir="/Users/ashraya/dms/"
infile=open(base_dir+"input_ready_data_v3.pickle","rb")
data_df=pkl.load(infile)
infile.close()
X=[]
Y=[]
for index,row in data_df.iterrows():
    row_to_add=[]
    for i in row['sequence_code']:
        row_to_add.append(i)
    row_to_add.append(row['fitness'])
    X.append(row_to_add)
    Y.append(row['inhibitor_code'])

X=np.array(X)
Y=np.array(Y)
# Y=Y.reshape(-1,1)

print("Data arrays are ready: "+str(X.shape)+","+str(Y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.30, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
print("Random forest classifier is being trained")
rf_classifier.fit(X_train, y_train)
print("Training done. Prediction starting")
y_pred = rf_classifier.predict(X_test)
print("Prediction done")
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

max((e.tree_.max_depth for e in rf_classifier.estimators_))

feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature Index': range(len(feature_importances)), 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv(base_dir+"one_hot_rf_feature_importances.csv",index=False)
feature_indices = np.arange(len(feature_importances))
plt.barh(feature_indices, feature_importances, align='center')
plt.yticks(feature_indices, feature_indices)  # Use numerical indices as tick labels
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title('Feature Importance Scores')

plt.gca().invert_yaxis()

