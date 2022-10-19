import torch
from torch.nn import Module, ModuleList
from torch_geometric.loader import DataLoader

from classes import AtomBondGraphFromMol, BondAngleGraphFromMol
from embed import GraphPairEmbedding
from split import ScaffoldSplit

KEYS = ['x_ab', 'x_ba', 'y']

class MoleculeDataLoader(Module):
    def __init__(self,
                 mol_list: list,
                 y_list: list[int],
                 config: dict={},
                 batch_size: int=1,
                 shuffle: bool=True,
                 device=torch.device('cpu'),
                 num_workers: int=4
        ):
        super(MoleculeDataLoader, self).__init__()

        self.embed_dim = config.get('embed_dim', 32)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.layer_num = config.get('layer_num', 8)
        self.readout = config.get('readout', 'mean')

        self.atom_feat_names = config['atom_feat_names']
        self.bond_feat_names = config['bond_feat_names']
        self.bond_float_names = config['bond_float_names']
        self.bond_angle_float_names = config['bond_angle_float_names']

        self.mol_list = mol_list
        self.y_list = y_list

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.num_workers = num_workers

        # self.split = ScaffoldSplit()

        self.embedding_list = ModuleList()

        for _ in range(len(mol_list)):
            # atom_bond_graph = AtomBondGraphFromMol(mol)
            # bond_angle_graph = BondAngleGraphFromMol(mol)
            self.embedding_list.append(GraphPairEmbedding(self.atom_feat_names, self.bond_feat_names, self.embed_dim))

    def forward(self) -> DataLoader:
        # train_indices, val_indices, test_indices = self.split.split(self.mol_list)

        # train_list, val_list, test_list = [], [], []

        graph_pair_list = []

        for i in range(len(self.mol_list)):
            print(i, self.mol_list[i].GetProp('_Name'))
            atom_bond_graph = AtomBondGraphFromMol(self.mol_list[i], self.y_list[i], self.device)
            bond_angle_graph = BondAngleGraphFromMol(self.mol_list[i], self.y_list[i], self.device)
            # print(bond_angle_graph.edge_features)
            graph_pair = self.embedding_list[i](atom_bond_graph, bond_angle_graph)

            graph_pair_list.append(graph_pair)

            # if i in train_indices:
            #     train_list.append(graph_pair)
            # elif i in val_indices:
            #     val_list.append(graph_pair)
            # elif i in test_indices:
            #     test_list.append(graph_pair)

        graph_pair_set = DataLoader(dataset=graph_pair_list,
                                    follow_batch=KEYS,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    num_workers=self.num_workers)

        # train_set  = DataLoader(dataset=train_list,
        #                         follow_batch=KEYS,
        #                         batch_size=self.batch_size,
        #                         shuffle=self.shuffle,
        #                         num_workers=self.num_workers)
        # val_set    = DataLoader(dataset=val_list,
        #                         follow_batch=KEYS,
        #                         batch_size=self.batch_size,
        #                         shuffle=self.shuffle,
        #                         num_workers=self.num_workers)
        # test_set   = DataLoader(dataset=test_list,
        #                         follow_batch=KEYS,
        #                         batch_size=self.batch_size,
        #                         shuffle=self.shuffle,
        #                         num_workers=self.num_workers)

        return graph_pair_set
