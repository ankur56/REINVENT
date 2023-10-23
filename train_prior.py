#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
#rdBase.DisableLog('rdApp.error')

def write_metrics_to_file(filename, epoch, metrics):
    with open(f"{filename}_epoch_{epoch}.txt", 'w') as f:
        for metric in metrics:
            f.write(str(metric) + '\n')

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc")

    # Create a Dataset from a SMILES file
    print('Loading data...')
    moldata = MolData("data/mols_filtered.smi", voc)
    print(f'Data Loaded. {len(moldata)} molecules available for training.')
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)
    print(f'Model Initialized with vocabulary size: {len(voc)}')
    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    all_losses = []
    all_validity = []
    for epoch in range(1, 6):
        print(f'\nStarting Epoch {epoch}/{5}')
        validity = []
        losses = []
        print('Epoch ' + str(epoch))
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()
            losses.append(loss)
            print(f'Batch Loss: {loss.item()}')

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Look at validities
            seqs, likelihood, _ = Prior.sample(128, max_length=150)
            valid = 0
            for i, seq in enumerate(seqs.cpu().numpy()):
                smile = voc.decode(seq)
                if Chem.MolFromSmiles(smile):
                    valid += 1
            validity.append(valid/len(seqs))
            print(f'Batch Validity: {validity[-1]}')

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                #tqdm.write("*" * 50)
                print("*" * 50, flush=True)
                #tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                #print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]), flush=True)
                print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()), flush=True)
                seqs, likelihood, _ = Prior.sample(128, max_length=150)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                #tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)), flush=True)
                #tqdm.write("*" * 50 + "\n")
                print("*" * 50 + "\n", flush=True)
                torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        #all_losses.append(losses)
        #all_validity.append(validity)

        #torch.save(all_losses, "data/Prior_losses")
        #torch.save(all_validity, "data/Prior_validities")
        # Store metrics to file after each epoch
        write_metrics_to_file("data/Prior_losses", epoch, losses)
        write_metrics_to_file("data/Prior_validities", epoch, validity)

        # Clear lists for the next epoch
        losses = []
        validity = []

        print("Step before saving final prior: "+str(epoch), flush=True)

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        print("Step after saving final prior at epoch: "+str(epoch), flush=True)
    #torch.save(all_losses, "data/Prior_losses")
    #torch.save(all_validity, "data/Prior_validities")

if __name__ == "__main__":
    pretrain()

# Things to look at
# Way to test RNN generation 
# Look at what figs they have for RNN validation 
# Compare to their dataset? 
