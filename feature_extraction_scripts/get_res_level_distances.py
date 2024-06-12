#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:13:11 2023

@author: ashraya
"""
import pandas as pd
opfile=open("cabo_res_level_distances.csv","w")
dist_df=pd.read_csv("/Users/ashraya/dms/cabo_distances_all.csv")
dist_df.sort_values(by=['protein residue number'])
for prot_resnum in dist_df['protein residue number'].unique():
    resnum_df=dist_df[dist_df['protein residue number']==prot_resnum]
    min_dist=min(resnum_df['distance'])
    min_dist_df=resnum_df[resnum_df['distance']==min_dist]
    opfile.write(min_dist_df['Ligand atom'].iloc[0]+","+min_dist_df['protein atom'].iloc[0]+","+str(min_dist_df['protein residue number'].iloc[0])+","+min_dist_df[' protein residue name'].iloc[0]+","+str(min_dist)+"\n")
opfile.close()