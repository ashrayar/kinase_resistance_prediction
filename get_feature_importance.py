#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:56:22 2024

@author: ashraya
"""

import xgboost as xgb
import pickle as pkl

feature_list = {"esm":['score','1','ESM LLR'],
                "dddg":['dddG','0','∆∆∆G'],
                "ddg_all":['ddG_all','-1','∆∆G'],
                "atp_distance":['distance','0','Residue->ATP Distance'],
                "volume_difference":['volume_difference','0','∆Volume'],
                #"inhibitor_type":['I','II','0','0','Inhibitor Type'],
                "inhibitor_weight":['molecular_weight','0','MWt'],
                "inhibitor_distance":['inhib_distance','0','Inhibitor distance'],
                "rf_score":['rfscore','0','RF Score'],
                "crystal_rmsf":['rmsf','0','MET Crystal RMSF'],
                "pocket_volume":['pocket_volume','0','Pocket Volume'],
                "hydrophobicity":['hydrophobicity_score','0','Hydrophobicity Score'],
                "polarity":['polarity_score','0','Polarity Score'],
                "residue_rmsd":['rmsd','0','Residue RMSD'],
                "ligand_rmsd":['ligrmsd','0','Ligand RMSD']
    
    }

def prepare_model_dict(features):
    model_dict=dict()
    monotone_array=[]
    feat_array=[]
    label_array=[]
    for feature in features:
        column_name,monotone,label = feature_list[feature]
        label_array.append(feature)
        monotone_array.append(monotone)
        feat_array.append(column_name)
    monotone_string="("
    monotone_string+=",".join(x for x in monotone_array)
    monotone_string+=")"
    dict_key="_".join(x for x in label_array)
    model_dict[dict_key]=[feat_array,monotone_string]
    return dict_key,model_dict

infile=open("models_of_interest.csv")
for line in infile:
    feature_set=line[0:-1].split(",")
    # print(feature_set)
    model_name,model_dict=prepare_model_dict(feature_set)
    trained_model = pkl.load(open("xgb_models/"+model_name+".pkl", "rb"))
    feature_important =trained_model.get_booster().get_score(importance_type='gain')
    print(line)
    print(feature_important)
infile.close()