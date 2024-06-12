#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:51:24 2024

@author: ashraya
"""

import pandas as pd
import numpy as np
import collections
# base_dir = "/wynton/home/fraserlab/aravikumar/dms/"
base_dir = "/Users/ashraya/dms/"

def my_rmsd(res1, res2):
    
    res1_sorted = collections.OrderedDict(sorted(res1.items()))
    res2_sorted = collections.OrderedDict(sorted(res2.items()))
    res1_coords = np.array(list(res1_sorted.values()))
    res2_coords = np.array(list(res2_sorted.values()))
    coor1 = res1_coords
    coor2 = res2_coords

    if coor1.shape != coor2.shape:
        raise ValueError("Coordinate shapes are not equivalent")
    diff = (coor1 - coor2).ravel()
    return np.sqrt(3 * np.inner(diff, diff) / diff.size)


map_df = pd.DataFrame(columns=['key','wt_lig','pred_lig'])
inhibitors = ["Crizo", "Tepo", "NVP", "Mere", "Savo", "Cabo", "Camp", "Gle", "Glu","A458"] # "Tiv","A458","DMSO"
for inhib in inhibitors:
    inhib_df = pd.read_csv(base_dir+inhib.lower()+"_atom_mapping.csv",header=None, names=['pred_lig','wt_lig'])
    inhib_array = [inhib] * len(inhib_df)
    inhib_df['key'] = inhib_array
    map_df=pd.concat([map_df,inhib_df],ignore_index=True)


# infile=open(base_dir+"umol_outputs/docking_done_no_issues")
# opfile=open(base_dir+"umol_outputs/ligand_rmsd.csv","w")
# opfile2=open(base_dir+"umol_outputs/ligand_rmsd_failed","w")
infile=['A131C_Crizo_wt_align_new.pdb']
ligand="Crizo"
for line1 in infile:
    # id,ligand=line1[0:-1].split(",")
    # print(line1[0:-1])
    alignfile=open(base_dir+"umol_outputs/"+line1)
    # alignfile=open(base_dir+"umol_outputs/"+id+"/"+ligand+"/"+id+"_wt_align/"+id+"_wt_align.pdb")
    wt_lig_atoms=dict()
    pred_lig_atoms=dict()
    flag=1
    for line in alignfile:
        if line.startswith("HETATM"):
            if line[17:20].strip() in ["GBL","CL"]:
                continue
            atom_name=line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            x=float(line[30:38])
            y=float(line[38:46])
            z=float(line[46:54])
            
            wt_lig_atoms[atom_name]=[x,y,z]
        elif line.startswith("ATOM"):
            if line[17:20].strip() == "UNL":
                atom_name=line[12:16].strip()
                if atom_name.startswith("H"):
                    continue
                full_atom_name=atom_name+line[6:11].strip()
                x=float(line[30:38])
                y=float(line[38:46])
                z=float(line[46:54])
                sub_map_df=map_df[map_df['key']==ligand]
                # sub_map_df=map_df[map_df['key']=="NVP"]
                try:
                    wt_atom_name = sub_map_df.loc[sub_map_df['pred_lig']==full_atom_name,'wt_lig'].iloc[0]
                except IndexError:
                    # opfile2.write(line1[0:-1])
                    # opfile.write(line1[0:-1]+",nan\n")
                    print(sub_map_df)
                    flag=0
                    break
                pred_lig_atoms[wt_atom_name]=[x,y,z]
    alignfile.close()
    if flag==1:
        print("wt atoms "+str(len(wt_lig_atoms)))
        print(wt_lig_atoms.keys())
        print("lig atoms"+str(len(pred_lig_atoms)))
        print(pred_lig_atoms.keys())
        lig_rmsd = my_rmsd(wt_lig_atoms, pred_lig_atoms)
        print(line1[0:-1]+str(round(lig_rmsd,2)))
        # opfile.write(line1[0:-1]+str(round(lig_rmsd,2))+"\n")
        print(lig_rmsd)
# opfile.close()
#infile.close()           
            
            