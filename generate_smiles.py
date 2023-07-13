# Needs to be run in the reinvent_2 env

# %%
# import matplotlib.pyplot as plt
import torch

from model import RNN
from data_structs import  Vocabulary
from utils import seq_to_smiles
from data_structs import write_smiles_to_file

# %%
# get trained RNN
# Read vocabulary from a file
voc = Vocabulary(init_from_file="data/Voc")

Prior = RNN(voc)

Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt'))


# Sample from Prior
seqs, likelihood, _ = Prior.sample(128, max_length=100)


# Translate to smiles
smiles = seq_to_smiles(seqs, voc)

write_smiles_to_file(smiles, "data/generated_mols.smi")
# %%



