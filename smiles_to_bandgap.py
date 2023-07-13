#!/usr/bin/env python3

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import os
import subprocess

def get_bandgap(smiles):
    # convert SMILES to 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    filename = "molecule"
    Chem.rdmolfiles.MolToXYZFile(mol, filename + '.xyz')
    print('file created')
    # Run xtb command
    cmd = 'xtb --verbose --grad -P 8 --chrg 0 --uhf 0 --namespace {0} -- {0}.xyz > {0}.out'.format(filename)
    subprocess.run(cmd, shell=True)

    # Parse output file for bandgap
    with open(filename + '.out') as f:
        lines = f.readlines()

    for line in lines:
        if 'HL-Gap' in line:
            return float(line.split()[-2])

    raise ValueError("Band gap not found in output file")

#print(get_bandgap('O'))
# weird smiles: print(get_bandgap('CC(C)[C@H]1[C@H]2C[C@@]1(C)O2'))




def get_bandgap_openbabel(smiles):
    # convert SMILES to 3D coordinates
    obConversion = pybel.ob.OBConversion()
    obConversion.SetInAndOutFormats("smi", "xyz")

    mol = pybel.readstring("smi", smiles)
    mol.make3D()
    
    filename = "molecule"
    xyz_output = obConversion.WriteString(mol.OBMol)

    with open(filename + '.xyz', 'w') as f:
        f.write(xyz_output)

    # Run xtb command
    cmd = 'xtb --verbose --grad -P 8 --chrg 0 --uhf 0 --namespace {0} -- {0}.xyz > {0}.out'.format(filename)
    subprocess.run(cmd, shell=True)

    # Parse output file for bandgap
    with open(filename + '.out') as f:
        lines = f.readlines()

    for line in lines:
        if 'HL-Gap' in line:
            return float(line.split()[-2])

    return -1
