from Bio import PDB
import numpy as np

def get_cb_atoms(structure):
    cb_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CB" in residue:
                    cb_atoms.append(residue["CB"])
    return cb_atoms

def find_residues_within_distance(structure, ligand_atoms, distance_threshold):
    cb_atoms = get_cb_atoms(structure)
    residues_within_distance = set()

    for cb_atom in cb_atoms:
        for ligand_atom in ligand_atoms:
            distance = np.linalg.norm(np.array(cb_atom.get_coord()) - np.array(ligand_atom.get_coord()))
            if distance <= distance_threshold:
                residues_within_distance.add(cb_atom.get_parent())
                break  # No need to check other ligand atoms for this CB atom

    return residues_within_distance

def main(pdb_file, ligand_chain_id, ligand_residue_id, ligand_residue_name, distance_threshold):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ligand_atoms = []
    for model in structure:
        for chain in model:
            if chain.id == ligand_chain_id:
                for residue in chain:
                    if residue.id[1] == ligand_residue_id and residue.resname == ligand_residue_name:
                        ligand_atoms.extend(residue)

    if not ligand_atoms:
        print("Ligand not found in the specified chain and residue position.")
        return

    residues_within_distance = find_residues_within_distance(structure, ligand_atoms, distance_threshold)

    print("Residues with CB atoms within {} Å from the ligand:".format(distance_threshold))
    for residue in residues_within_distance:
        print(residue.id[1])

if __name__ == "__main__":
    pdb_file = "/Users/ashraya/dms/wt_reference_structures/3dkc_dmso.pdb"
    ligand_chain_id = "A"
    ligand_residue_id = 1
    ligand_residue_name = "ATP"
    distance_threshold = 10.0

    main(pdb_file, ligand_chain_id, ligand_residue_id, ligand_residue_name, distance_threshold)