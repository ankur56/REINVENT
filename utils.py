import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
#import pickle
import pickle5 as pickle



def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#

# Score is between 1 (easy to synthesize) and 10 (difficult to synthesize) 
# Above 6 are classified as difficult to synthesize
def sa_score(smiles):
    sa_scores = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            sa_scores.append(sascorer.calculateScore(mol))

    return sa_scores

def percentage_easy_sa(smiles):
    sa_scores = sa_score(smiles)
    return np.sum(np.array(sa_scores) < 7)/len(sa_scores)

def pickle_to_data(filename):
    with open(filename, "rb") as handle:
        qm9_data = pickle.load(handle)

    # Convert the keys and values of the dictionary into separate lists
    smiles_list = list(qm9_data.keys())
    property_list = list(qm9_data.values())

    # Extract the bandgap as a separate list
    bandgap = [prop[3] for prop in property_list]

    return smiles_list, bandgap

# Percentage of smiles in a list that are 
def percentage_unique(smiles):
    # import qm9 dataset
    train_smiles_list, train_bandgap = pickle_to_data("data//qm9_key_smiles_1_full_train_data.pickle")
    holdout_smiles_list, holdout_bandgap = pickle_to_data("data/qm9_key_smiles_1_holdout_data.pickle")
    full_qm9_smiles_list = train_smiles_list + holdout_smiles_list

    unique = [smile in full_qm9_smiles_list for smile in smiles]
    pcnt_unique = (len(unique)-sum(unique))/len(unique)
    return pcnt_unique