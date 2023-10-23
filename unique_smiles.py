#!/usr/bin/env python

def identify_unique_and_duplicate_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file.readlines()]
    
    unique_smiles = set()
    duplicate_smiles = set()
    for smi in smiles_list:
        if smi in unique_smiles:
            duplicate_smiles.add(smi)
        else:
            unique_smiles.add(smi)
    
    # Removing duplicates from the unique_smiles set
    unique_smiles = unique_smiles - duplicate_smiles
    
    return unique_smiles, duplicate_smiles

# Usage
file_path = 'absent_smiles.smi'
unique_smiles, duplicate_smiles = identify_unique_and_duplicate_smiles(file_path)

print(f"Number of unique SMILES: {len(unique_smiles)}")
print(f"Number of duplicate SMILES: {len(duplicate_smiles)}")

# If you want to print or save the SMILES:
# print("Unique SMILES:")
# for smi in unique_smiles:
#     print(smi)

# print("\nDuplicate SMILES:")
# for smi in duplicate_smiles:
#     print(smi)

