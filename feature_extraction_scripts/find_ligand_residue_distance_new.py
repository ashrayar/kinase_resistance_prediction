#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:33:39 2024

@author: ashraya
"""
from Bio import PDB
import numpy as np
import sys

def calculate_distance(atom1, atom2):
    return np.linalg.norm(np.array(atom1.get_coord()) - np.array(atom2.get_coord()))

def find_shortest_distance(protein_atoms, ligand_atoms):
    shortest_distances = {}
    
    for residue_atom in protein_atoms:
        # print("residue_atom:"+str(residue_atom.get_name()))
        residue_id = residue_atom.get_parent().id
        residue_key = f"{residue_id[1]}"
        if residue_key not in shortest_distances:
            shortest_distances[residue_key] = float('inf')

        for ligand_atom in ligand_atoms:
            distance = calculate_distance(residue_atom, ligand_atom)
            if residue_atom.get_name()[0] == "H" or ligand_atom.get_name()[0] == "H":
                continue
            if distance < shortest_distances[residue_key]:
                shortest_distances[residue_key] = distance

    return shortest_distances

def main(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file_path)

    protein_atoms = []
    ligand_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if chain.id ==" ":
                    ligand_atoms.extend(residue)
                    # Protein residue
                elif chain.id == "A":
                    # Ligand residue
                    protein_atoms.extend(residue)

    for ligand_atom in ligand_atoms:
        print(ligand_atom.get_name())
    shortest_distances = find_shortest_distance(protein_atoms, ligand_atoms)

    # opfile=open("/wynton/home/fraserlab/aravikumar/dms/umol_outputs/"+pos_mut+"/"+inhib+"/"+pos_mut+"_docked_inhib_distance.csv","w")
    opfile=open("distance_test.csv","w")
    for residue_key, distance in shortest_distances.items():
      opfile.write(f"{residue_key},{distance:.2f}\n")
    opfile.close()

if __name__ == "__main__":
    #pos_mut = sys.argv[1]
    #inhib = sys.argv[2]
    # pdb_file_path = "/wynton/home/fraserlab/aravikumar/dms/umol_outputs/"+pos_mut+"/"+inhib+"/"+pos_mut+"_docked_complex.pdb"
    pdb_file_path = "umol_outputs/V98K_docked_complex.pdb"
    main(pdb_file_path)
