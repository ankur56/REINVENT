#!/usr/bin/env python3

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import os
import subprocess
import hashlib
import logging
import time

#logging.basicConfig(filename='smiles_gap_log.txt', level=logging.DEBUG)

def setup_custom_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger that logs info to specified log file."""
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler that logs debug level and higher messages to a file
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    # Prevent logging from propagating to the root logger
    logger.propagate = False
    
    return logger

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

    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
    except Exception as e:
        print(f"Error: {e}")
        return -1
    
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

def get_bandgap_unique(smiles, xtb_run):
    # convert SMILES to 3D coordinates
    obConversion = pybel.ob.OBConversion()
    obConversion.SetInAndOutFormats("smi", "xyz")

    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    # Create a SHA-1 hash of the SMILES string
    hash_object = hashlib.sha1(smiles.encode())
    hex_dig = hash_object.hexdigest()
    
    filename = hex_dig
    xyz_output = obConversion.WriteString(mol.OBMol)

    #xtb_dir = 'xtb_data_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    xtb_dir = 'xtb_data_' + xtb_run
    if not os.path.exists(xtb_dir):
        os.makedirs(xtb_dir)
    file_path = os.path.join(xtb_dir, filename)
    with open(file_path + '.xyz', 'w') as f:
        f.write(xyz_output)

    # Run xtb command
    #cmd = 'xtb --verbose -P 4 --chrg 0 --uhf 0 --ohess --opt extreme --namespace {0} -- {0}.xyz > {0}.out'.format(filename)
    cmd = 'xtb --verbose -P 4 --chrg 0 --uhf 0 --opt tight --namespace {0} -- {0}.xyz > {0}.out'.format(file_path)
    subprocess.run(cmd, shell=True)

    # Use grep command to find the line with HOMO-LUMO GAP and extract the value
    grep_cmd = "grep 'HOMO-LUMO GAP' {0}.out | awk '{{print $4}}'".format(file_path)
    gap = subprocess.check_output(grep_cmd, shell=True)

    # if gap == 2.0 within a threshold, perform DFT
    # Remove unnecessary files
    files_to_remove = [file_path + ext for ext in ['.wbo', '.vibspectrum', '.xtbopt.log', '.charges', '.hessian', '.g98.out', '.xtbrestart', '.xtbtopo.mol']]
    for file in files_to_remove:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

    #if gap:
    #    logger = setup_custom_logger('bandgap_logger', 'bandgap.log')
    #    logger.info(f"SMILES: {smiles}, GAP: {float(gap)}")
        
    if isinstance(float(gap), float):
        return float(gap)
    else:
        print('Error in SMILE: '+smiles+' '+filename, flush=True)
        return -1

if __name__ == "__main__":
    get_bandgap_unique('C')

