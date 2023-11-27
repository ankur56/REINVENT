#!/usr/bin/env python3

import sys
import selfies as sf

def read_selfies_from_file(fname):
    """Reads a SELFIES file and returns a list of SELFIES strings."""
    with open(fname, 'r') as f:
        selfies_list = [line.strip() for line in f if line.strip()]
    return selfies_list

def construct_vocabulary(selfies_list):
    """Constructs a vocabulary from a list of SELFIES."""
    add_chars = set()
    for selfies in selfies_list:
        chars = list(sf.split_selfies(selfies))
        add_chars.update(chars)
    
    print(f"Number of characters: {len(add_chars)}")
    with open('data/Voc_selfies.txt', 'w') as f:
        for char in sorted(add_chars):
            f.write(char + "\n")
    return add_chars

if __name__ == "__main__":
    selfies_file = sys.argv[1]
    print("Reading SELFIES...")
    selfies_list = read_selfies_from_file(selfies_file)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(selfies_list)



