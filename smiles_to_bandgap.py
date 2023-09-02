#!/usr/bin/env python3

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import os
import subprocess
import hashlib

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

def get_bandgap_unique(smiles):
    # convert SMILES to 3D coordinates
    obConversion = pybel.ob.OBConversion()
    obConversion.SetInAndOutFormats("smi", "xyz")

    mol = pybel.readstring("smi", smiles)
    mol.make3D()

    # Create a SHA-1 hash of the SMILES string
    hash_object = hashlib.sha1(smiles.encode())
    hex_dig = hash_object.hexdigest()
    
    filename = hex_dig
    xyz_output = obConversion.WriteString(mol.OBMol)

    with open(filename + '.xyz', 'w') as f:
        f.write(xyz_output)

    # Run xtb command
    cmd = 'xtb --verbose -P 8 --chrg 0 --uhf 0 --ohess --opt extreme --namespace {0} -- {0}.xyz > {0}.out'.format(filename)
    subprocess.run(cmd, shell=True)

    # Use grep command to find the line with HOMO-LUMO GAP and extract the value
    grep_cmd = "grep 'HOMO-LUMO GAP' {0}.out | tail -1 | awk '{{print $4}}'".format(filename)
    gap = subprocess.check_output(grep_cmd, shell=True)

    # Remove unnecessary files
    files_to_remove = [filename + ext for ext in ['.wbo', '.vibspectrum', '.xtbopt.log', '.charges', '.hessian', '.g98.out', '.xtbrestart', '.xtbtopo.mol']]
    for file in files_to_remove:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
    
    if not gap:
        return -1

    return float(gap)


