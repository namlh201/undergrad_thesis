from __future__ import annotations

from collections import defaultdict

import numpy as np

from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles: str, include_chirality: bool=False):
    """return scaffold string of target molecule"""
    # mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

class ScaffoldSplit():
    # Src: https://github.com/chainer/chainer-chemistry/blob/master/chainer_chemistry/smiles_list/splitters/scaffold_splitter.py
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def split(self, smiles_list: list, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns index list of train, valid, test set
        np.testing.assert_almost_equal(self.train_size + self.val_size + self.test_size,
                                       1.)

        seed = kwargs.get('seed', None)
        include_chirality = kwargs.get('include_chirality', True)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for i, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality)
            scaffolds[scaffold].append(i)

        scaffold_sets = rng.permutation(list(scaffolds.values()))

        n_total_valid = int(np.floor(self.val_size * len(smiles_list)))
        n_total_test = int(np.floor(self.test_size * len(smiles_list)))

        train_index = []
        valid_index = []
        test_index = []

        for scaffold_set in scaffold_sets:
            if len(valid_index) + len(scaffold_set) <= n_total_valid:
                valid_index.extend(scaffold_set)
            elif len(test_index) + len(scaffold_set) <= n_total_test:
                test_index.extend(scaffold_set)
            else:
                train_index.extend(scaffold_set)

        return train_index, valid_index, test_index
