# %%
from data_structs import canonicalize_smiles_from_file
from rdkit import Chem

# %%
# Import RNN output
output_smiles = canonicalize_smiles_from_file('data/generated_mols.smi')

# %%
# Inspect validity
i = 0
for smile in output_smiles:
    if Chem.MolFromSmiles(smile):
        i += 1
frac_validity = i / len(output_smiles)
print('Fraction of validity: ' + str(frac_validity))

# %%
# Test bandgap computation
# for now store in smiles_to_bangap.py
from smiles_to_bandgap import get_bandgap

example_smiles = output_smiles[0]

bandgap = get_bandgap(example_smiles[0])
print(bandgap)


# %%


# %%



