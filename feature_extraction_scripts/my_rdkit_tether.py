#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:26:09 2024

@author: ashraya
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import sys

def tether_mol_v1(tether, mol):
    w=Chem.SDWriter(sys.argv[3])
    ratioThreshold=0.20
    mcs = rdFMCS.FindMCS([tether, mol], threshold=0.9, completeRingsOnly=True, matchValences=True)
    print('smarts:', mcs.smartsString)
    patt = Chem.MolFromSmarts(mcs.smartsString,mergeHs=True)
    replaced = AllChem.ReplaceSidechains(tether, mcs.queryMol)
    core = AllChem.DeleteSubstructs(replaced, Chem.MolFromSmiles('*'))
    core.UpdatePropertyCache()
    molh = Chem.AddHs(mol)
    #AllChem.ConstrainedEmbed(molh, core)
    AllChem.ConstrainedEmbed(molh, core,enforceChirality = False) #For ATP
    rms=float(molh.GetProp('EmbedRMS'))
    print('Common core: ',Chem.MolToSmiles(core),'  - RMSD:',rms)
    matchratio=float(core.GetNumAtoms())/float(tether.GetNumAtoms())
    tethered_atom_ids=molh.GetSubstructMatches(patt)
    if tethered_atom_ids and matchratio>ratioThreshold : 
        t=tethered_atom_ids[0]
        t1=map(lambda x:x+1, list(t))
        ta=','.join(str(el) for el in t1)
        nm=Chem.AddHs(molh, addCoords=True)	#create a new 3D molecule  and add the TETHERED ATOMS property
        nm.SetProp('TETHERED ATOMS',ta)
        w.write(nm)
    return molh


#tether = Chem.MolFromMolFile(sys.argv[1],sanitize=False)
tether = Chem.MolFromMolFile(sys.argv[1],sanitize=True)
ligand = Chem.MolFromMolFile(sys.argv[2])
mymol = tether_mol_v1(tether, ligand)
