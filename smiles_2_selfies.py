#!/usr/bin/env python3

import argparse
import logging
from selfies import encoder
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

def smiles_to_selfies(smiles_string):
    try:
        selfies_string = encoder(smiles_string)
        return selfies_string
    except Exception as e:
        logging.error(f"Error converting SMILES '{smiles_string}': {e}")
        return None

def main(args):
    # Check if input file exists
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            smiles_list = f.read().splitlines()
    except FileNotFoundError:
        logging.error(f"Input file {args.input_file} not found.")
        return

    # Parallelize the conversion using multiprocessing
    with Pool(min(cpu_count(), 16)) as pool:  # Using min to ensure no more than 4 processors
        selfies_list = list(tqdm(pool.imap(smiles_to_selfies, smiles_list), total=len(smiles_list)))

    # Filter out None values if any
    selfies_list = [s for s in selfies_list if s is not None]

    # Write the SELFIES to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for selfies_string in selfies_list:
            f.write(selfies_string + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts SMILES strings to SELFIES.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing SMILES strings.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file to store the SELFIES strings.')

    args = parser.parse_args()
    main(args)
