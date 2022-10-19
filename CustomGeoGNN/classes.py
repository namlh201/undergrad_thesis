from __future__ import annotations

import torch
from torch import Tensor
# import torch.nn as nn
# import torch.nn.functional as F

from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .featurize import featurize_atom, featurize_atom_mass, featurize_bond, featurize_bond_length, featurize_bond_angle

class GraphFromMol():
    def __init__(self, mol: Mol):#, device=torch.device('cpu')):
        self.mol = mol
        # self.device = device

    def get_node_feat_names(self):
        return list(self.node_features.keys())

    def get_edge_feat_names(self):
        return list(self.edge_features.keys())

    def _get_adjacency_matrix(self) -> torch.Tensor:
        return GetAdjacencyMatrix(self.mol)

class AtomBondGraphFromMol(GraphFromMol):
    def __init__(self, mol: Mol):#, y: float | None=None):#, device=torch.device('cpu')):
        super(AtomBondGraphFromMol, self).__init__(mol)#, device)

        self.node_features = featurize_atom(mol)
        self.node_features['mass'] = featurize_atom_mass(mol)['mass']

        self.edge_features = featurize_bond(mol)
        self.edge_features['length'] = featurize_bond_length(mol)['length']

        self.num_nodes = len(self.node_features[list(self.node_features.keys())[0]])
        self.num_edges = len(self.edge_features[list(self.edge_features.keys())[0]])

        self.adj_mat = self._get_atom_bond_graph()

        self.edge_index = self.adj_mat.to_sparse_coo().indices()

        assert self.num_nodes == int(self.edge_index.max()) + 1

        self.node_edge_idx_map = self._get_bond_idx_map()

    def _get_bond_idx_map(self) -> dict[tuple, int]:
        # Get a tensor that maps from node (ith, jth) to edge kth
        # Each entry: [node_i_idx, node_j_idx]: edge_idx
        bond_idx_map = {}

        for bond in self.mol.GetBonds():
            bond_idx = bond.GetIdx()
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            bond_idx_map[(begin_atom_idx, end_atom_idx)] = bond_idx
            bond_idx_map[(end_atom_idx, begin_atom_idx)] = bond_idx

            # bond_idx_map[(begin_atom_idx, end_atom_idx)] = bond_idx
            # bond_idx_map[(end_atom_idx, begin_atom_idx)] = bond_idx


        # for bond in self.mol.GetBonds():
        #     bond_idx_list.append(bond.GetIdx())

        return bond_idx_map

    def _get_atom_bond_graph(self) -> Tensor:
        return torch.tensor(self._get_adjacency_matrix())

class BondAngleGraphFromMol(GraphFromMol):
    def __init__(self, mol: Mol):#, y: float | None=None):#, device=torch.device('cpu')):
        super(BondAngleGraphFromMol, self).__init__(mol)#, device)

        self.node_features = featurize_bond(mol)
        self.node_features['length'] = featurize_bond_length(mol)['length']

        self.edge_features = featurize_bond_angle(mol)

        self.num_nodes = len(self.node_features[list(self.node_features.keys())[0]])
        self.num_edges = len(self.edge_features[list(self.edge_features.keys())[0]])

        self.adj_mat = self._get_bond_angle_graph()

        self.edge_index = self.adj_mat.to_sparse_coo().indices()



        self.node_edge_idx_map = self._get_angle_idx_map()

    def _get_angle_idx_map(self) -> dict[tuple, int]:
        # Get a tensor that maps from node (ith, jth) to edge kth
        # Each entry: [node_i_idx, node_j_idx]: edge_idx
        angle_idx_map = {}

        angle_idx = 0

        bonds = self.mol.GetBonds()
        length = self.mol.GetNumBonds()

        bond_couples = []

        # graph = torch.zeros((nrows, ncols), dtype=torch.long)

        for i in range(length):
            for j in range(length):
                if i == j: continue

                if (i, j) in bond_couples or (j, i) in bond_couples: continue

                if bonds[i].GetBeginAtomIdx() == bonds[j].GetBeginAtomIdx() \
                or bonds[i].GetBeginAtomIdx() == bonds[j].GetEndAtomIdx() \
                or bonds[i].GetEndAtomIdx()   == bonds[j].GetBeginAtomIdx() \
                or bonds[i].GetEndAtomIdx()   == bonds[j].GetEndAtomIdx():
                    angle_idx_map[(i, j)] = angle_idx
                    angle_idx_map[(j, i)] = angle_idx
                    # angle_idx_map[(i, j)] = angle_idx
                    # angle_idx_map[(j, i)] = angle_idx

                    angle_idx += 1

                bond_couples.append((i, j))
                bond_couples.append((j, i))

        return angle_idx_map

    def _get_bond_angle_graph(self) -> Tensor:
        bonds = self.mol.GetBonds()

        nrows = ncols = len(bonds)

        bond_couples = []

        graph = torch.zeros((nrows, ncols), dtype=torch.long)

        for i in range(nrows):
            for j in range(ncols):
                if i == j: continue

                if (i, j) in bond_couples or (j, i) in bond_couples: continue

                if bonds[i].GetBeginAtomIdx() == bonds[j].GetBeginAtomIdx() \
                or bonds[i].GetBeginAtomIdx() == bonds[j].GetEndAtomIdx() \
                or bonds[i].GetEndAtomIdx()   == bonds[j].GetBeginAtomIdx() \
                or bonds[i].GetEndAtomIdx()   == bonds[j].GetEndAtomIdx():
                    graph[i][j] = 1
                    graph[j][i] = 1

                bond_couples.append((i, j))
                bond_couples.append((j, i))

        return graph