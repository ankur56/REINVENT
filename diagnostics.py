
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

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
def sa_score(SMILE):
    mol = Chem.MolFromSmiles('NC(=O)c1ccccc1')
    s = sascorer.calculateScore(mol)

    return s