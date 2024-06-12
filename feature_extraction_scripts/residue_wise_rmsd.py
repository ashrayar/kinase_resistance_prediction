#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:55:43 2024

@author: ashraya
"""

from qfit import Structure
import numpy as np
def my_rmsd(res1, res2):
        coor1 = res1.coor
        coor2 = res2.coor

        if coor1.shape != coor2.shape:
            raise ValueError("Coordinate shapes are not equivalent")
        diff = (coor1 - coor2).ravel()
        return np.sqrt(3 * np.inner(diff, diff) / diff.size)

base_dir="/wynton/home/fraserlab/aravikumar/dms/umol_outputs/"
infile=open(base_dir+"docking_done_no_issues")
count=1

for line in infile:
    print(line[0:-1])
    if count < 2875:
        count+=1
        continue
    count+=1
    logfile=open(base_dir+"rmsd_calc_running","a")
    logfile.write(line)
    lineparts=line[0:-1].split(",")
    logfile.close()
    align_prot=Structure.fromfile(base_dir+lineparts[0]+"/"+lineparts[1]+"/"+lineparts[0]+"_wt_align/"+lineparts[0]+"_wt_align.pdb")
    ref_struct=align_prot.extract("chain","A")
    # pred_struct=align_prot.extract("chain","B").extract("resn","UNL","!=")
    pred_struct=align_prot.extract("chain","B").extract("resn","UNK","!=") #For ATP
    resnum=int(lineparts[0][1:-1])
    pred_residue=pred_struct.extract("resi",resnum).extract("name",["CA","C","O","N"])
    ref_residue=ref_struct.extract("resi",resnum).extract("name",["CA","C","O","N"])
    print(str(len(pred_residue.resi))+","+str(len(ref_residue.resi)))
    if len(ref_residue.resi) < 4 or len(pred_residue.resi)<4:
        opfile=open(base_dir+"rmsd_from_wt.csv","a")
        opfile.write(line[0:-1]+",nan\n")
        opfile.close()
        continue
    elif len(ref_residue.resi) != len(pred_residue.resi):
        opfile=open(base_dir+"rmsd_from_wt.csv","a")
        opfile.write(line[0:-1]+",nan\n")
        opfile.close()
        continue
    rmsd=my_rmsd(pred_residue,ref_residue)
    opfile=open(base_dir+"rmsd_from_wt.csv","a")
    opfile.write(line[0:-1]+","+str(round(rmsd,3))+"\n")
    opfile.close()

infile.close()
opfile.close()