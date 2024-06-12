#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:13:23 2023

@author: ashraya
"""


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
infile=open("/Users/ashraya/dms/all_seq_esm_embeddings.csv")
Xs=[]
ys=[]
for line in infile:
    lineparts=line[0:-1].split(",")
    embeds=[]
    for i in range(1,len(lineparts)):
        embeds.append(float(lineparts[i]))
    Xs.append(embeds)
    fitfile=open("/Users/ashraya/dms/all_fitness_data_final.csv")
    x=fitfile.readline()
    for line1 in fitfile:
        line1parts=line1.split(",")
        if line1parts[0]==lineparts[0]:
            ys.append(float(line1parts[1]))
            break
    fitfile.close()
infile.close()
Xs_np=np.array(Xs)
ys_np=np.array(ys)

categories=[]
for y in ys:
    if y > -4.0 and y < -2.0:
        categories.append("between -3.0 and -2.0")
    elif y >= -2.0 and y < -1.0:
        categories.append("between -2.0 and -1.0")
    elif y >= -1.0:
        categories.append("above -1.0")

num_pca_components = 287
pca = PCA(num_pca_components)
Xs_train_pca = pca.fit_transform(Xs_np)
df = pd.DataFrame(dict(Xs=Xs_train_pca[:,0], ys=Xs_train_pca[:,1], fitness_score=categories))

relpt=sns.relplot(data=df, x='Xs', y='ys', hue='fitness_score')
relpt.set(xlabel ="PCA first principal component", ylabel = "PCA second principal component")
with sns.plotting_context(rc={"legend.fontsize":14}):
    relpt = sns.relplot(data=df, x='Xs', y='ys', hue='fitness_score',s=40)
    plt.xlabel("First principal component",fontsize = 18)
    plt.ylabel("Second principal component",fontsize = 18)



# fig_dims = (7, 6)
# fig, ax = plt.subplots(figsize=fig_dims)
# sc = ax.scatter(Xs_train_pca[:,0], Xs_train_pca[:,1], c=ys_np, marker='.')
# ax.set_xlabel('PCA first principal component')
# ax.set_ylabel('PCA second principal component')
# plt.colorbar(sc, label='Variant Effect')
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.savefig("ESM_dmso_pca_correct.pdf",format="pdf",bbox_inches="tight")
plt.savefig("ESM_dmso_pca_correct.png",format="png",dpi = 300, bbox_inches="tight")


#ESM + fingerprint
embed_finger_df = pd.read_csv("/Users/ashraya/dms/esm_embed_fingerprint_with_fitness.csv",header = 0)
temp_cols=embed_finger_df.columns.tolist()
new_cols=temp_cols[0:1281] + temp_cols[1283:] + [temp_cols[1282]] + [temp_cols[1281]]
embed_finger_df=embed_finger_df[new_cols]
no_dmso_df = embed_finger_df[embed_finger_df.key != "DMSO"]
sub_embed_df=no_dmso_df.iloc[:,1:2306]

num_pca_components = 5
pca = PCA(num_pca_components)
Xs_train_pca = pca.fit_transform(sub_embed_df.iloc[:,0:2305])
ys = np.array(sub_embed_df.iloc[:,2304:2305]).flatten()
fig_dims = (7, 6)
fig, ax = plt.subplots(figsize=fig_dims)
mol_names=np.array(no_dmso_df['key']).flatten()
sc = ax.scatter(Xs_train_pca[:,0], Xs_train_pca[:,1], c=embed_finger_df['key'], marker='.')
df = pd.DataFrame(dict(Xs=Xs_train_pca[:,0], ys=Xs_train_pca[:,1], inhibitor=mol_names))
with sns.plotting_context(rc={"legend.fontsize":14}):
    relpt = sns.relplot(data=df, x='Xs', y='ys', hue='inhibitor',palette = 'tab20',s=70)
    plt.xlabel("First principal component",fontsize = 18)
    plt.ylabel("Second principal component",fontsize = 18)
plt.savefig("ESM_fingerprint_pca.pdf",format="pdf",bbox_inches="tight")
plt.savefig("ESM_fingerprint_pca.png",format="png",dpi = 300,bbox_inches="tight")

#ESM + distance

embed_dist_df = pd.read_csv("/Users/ashraya/dms/esm_embed_distances_with_fitness.csv",header = 0)
no_dmso_df = embed_dist_df[embed_dist_df.key != "DMSO"]
sub_embed_df = no_dmso_df.iloc[:,1:1568]
normal_sub_embed_df=(sub_embed_df-sub_embed_df.mean())/sub_embed_df.std()
num_pca_components = 5
pca = PCA(num_pca_components)
Xs_train_pca = pca.fit_transform(normal_sub_embed_df)
ys=np.array(embed_dist_df['mean']).flatten()
fig_dims = (7, 6)
fig, ax = plt.subplots(figsize=fig_dims)
sc = ax.scatter(Xs_train_pca[:,0], Xs_train_pca[:,1], c=ys, marker='.')
mol_names=np.array(no_dmso_df['key']).flatten()
df = pd.DataFrame(dict(Xs=Xs_train_pca[:,0], ys=Xs_train_pca[:,1], inhibitor=mol_names))
with sns.plotting_context(rc={"legend.fontsize":14}):
    relpt = sns.relplot(data=df, x='Xs', y='ys', hue='inhibitor',palette = 'tab20',s=70)
    plt.xlabel("First principal component",fontsize = 18)
    plt.ylabel("Second principal component",fontsize = 18)
plt.savefig("ESM_distance_pca_no_dmso.pdf",format="pdf",bbox_inches="tight")
plt.savefig("ESM_distance_pca_no_dmso.png",format="png",bbox_inches="tight")