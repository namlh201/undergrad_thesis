from __future__ import annotations

# from typing import Optional
import torch
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList, Sequential, ReLU, GELU, SmoothL1Loss, MSELoss
from torch_geometric.nn import global_mean_pool, global_add_pool, Linear
from tqdm import tqdm

from .embed import GraphPairEmbeddingBatch
from .module.block import GeoGNNBlock, TransformerBlock
from .module.loss_fn import LaplacianEigenvectorLoss, DynamicWeightAverageLoss, UncertaintyLoss, StdSmoothL1Loss
from .module.utils import UnitNorm
from .utils.graph_utils import GraphUtils

class GeoGNNModel(Module):
    def __init__(self, config={}):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = config['model']['embed_dim']
        self.dropout_rate = config['model']['dropout_rate']
        self.layer_num = config['model']['layer_num']
        self.readout = config['model']['readout']
        self.batch_size = config['model']['batch_size']

        self.walk_length = config['pe']['walk_length']

        self.atom_feat_names = config['atom_feat_names']
        self.bond_feat_names = config['bond_feat_names']
        # self.bond_float_names = config['bond_float_names']
        # self.bond_angle_float_names = config['bond_angle_float_names']

        self.graph_pair_embedding = GraphPairEmbeddingBatch(
            self.atom_feat_names,
            self.bond_feat_names,
            self.walk_length,
            self.batch_size,
            self.embed_dim
        )

        self.atom_bond_block_list = ModuleList()
        self.bond_angle_block_list = ModuleList()

        for layer_idx in range(self.layer_num):
            self.atom_bond_block_list.append(
                GeoGNNBlock(
                    self.embed_dim,
                    self.dropout_rate,
                    last_act=(layer_idx != self.layer_num - 1)
                )
            )
            self.bond_angle_block_list.append(
                GeoGNNBlock(
                    self.embed_dim,
                    self.dropout_rate,
                    last_act=(layer_idx != self.layer_num - 1)
                )
            )

        self.transformer_block = TransformerBlock(self.embed_dim, num_layers=2, heads=4, dropout_rate=self.dropout_rate)

        self.pe_out = Sequential(
            Linear(self.embed_dim, self.walk_length, weight_initializer='glorot'),
            UnitNorm(dim=0) # normalize
        )

        if self.readout == 'mean':
            self.graph_pool = global_mean_pool
        elif self.readout == 'sum':
            self.graph_pool = global_add_pool

    @property
    def device(self):
        return next(self.parameters()).device

    # def forward(self, graph_pair) -> tuple[Tensor, Tensor, Tensor]:
    def forward(self, atom_bond_graph_list, bond_angle_graph_list) -> tuple[Tensor, Tensor, Tensor]:
        edge_index_ab, x_ab, edge_attr_ab, edge_map_ab, adj_mat_ab, pe_ab, batch_index_ab, \
        edge_index_ba, x_ba, edge_attr_ba, edge_map_ba, adj_mat_ba, \
        batch_list = self.graph_pair_embedding(atom_bond_graph_list, bond_angle_graph_list)

        atom_node_hidden_feat = x_ab.to(self.device)
        bond_edge_hidden_feat = edge_attr_ab.to(self.device)
        bond_edge_index = edge_index_ab.to(self.device)

        bond_node_hidden_feat = x_ba.to(self.device)
        angle_edge_hidden_feat = edge_attr_ba.to(self.device)
        angle_edge_index = edge_index_ba.to(self.device)

        node_hidden_list = []
        edge_hidden_list = []
        pe_hidden_list = []

        # print('pe_ab', pe_ab)

        # for layer_idx in tqdm(range(self.layer_num), desc='Learning from embed'):
        for layer_idx in range(self.layer_num):
            # print('Atom Bond Block #{}'.format(layer_idx + 1))
            # print('bond_edge_index', bond_edge_index, bond_edge_index.shape, bond_edge_index.max())
            # print('atom_node_hidden_feat', atom_node_hidden_feat, atom_node_hidden_feat.shape, atom_node_hidden_feat.size(0))
            # print('pe_ab', pe_ab, pe_ab.shape, pe_ab.size(0))
            atom_node_hidden_feat, pe_ab = self.atom_bond_block_list[layer_idx](
                x=atom_node_hidden_feat,
                edge_index=bond_edge_index,
                edge_attr=bond_edge_hidden_feat,
                pe=pe_ab
            )
            # print(pe_ab)

            # curr_edge_hidden_feat = atom_bond_graph['edge_attr']
            # curr_angle_hidden_feat = bond_angle_graph['edge_attr']

            # print('Bond Angle Block #{}'.format(layer_idx + 1))
            bond_node_hidden_feat = self.bond_angle_block_list[layer_idx](
                x=bond_node_hidden_feat,
                edge_index=angle_edge_index,
                edge_attr=angle_edge_hidden_feat
            )

            node_hidden_list.append(atom_node_hidden_feat)
            edge_hidden_list.append(bond_node_hidden_feat)
            pe_hidden_list.append(pe_ab)
            
            # print(bond_node_hidden_feat)

            # bond_edge_hidden_feat = self.lin_bond_edge(bond_node_hidden_feat.T).T

            # bond_edge_hidden_feat = double_edge_feature(bond_edge_index, bond_node_hidden_feat)
            # bond_edge_hidden_feat = atom_bond_graph._get_edge_attribute_as_edge_matrix(bond_node_hidden_feat)
            bond_edge_hidden_feat = GraphUtils.get_edge_attribute_as_edge_matrix(
                edge_index_ab,
                edge_map_ab,
                bond_node_hidden_feat
            )

        node_repr_feat = node_hidden_list[-1]
        edge_repr_feat = edge_hidden_list[-1]
        pe_repr_feat = pe_hidden_list[-1]

        node_feat = self.transformer_block(node_repr_feat, pe_repr_feat, bond_edge_index)
        pe_repr_feat = self.pe_out(pe_repr_feat)

        # node_feat = node_repr_feat

        if node_feat.size(0) < node_repr_feat.size(0):
            node_feat = torch.cat((node_feat.T.view(-1, node_feat.size(0)), torch.mean(node_feat, dim=1, keepdim=True).view(-1, 1)), dim=1).T
        node_feat = node_repr_feat + node_feat
        # print(node_feat)
        # print(pe_repr_feat)

        # node_feat = []
        # for hidden_feat in node_hidden_list:
        #     hidden_feat = hidden_feat + node_tf_feat

        # graph_repr_feat = torch.cat(node_feat, dim=1)

        # batch_list = graph_pair.y_batch

        # graph_repr_feat = global_mean_pool(node_repr_feat, None)

        # node_feat = node_hidden_list[-4:]
        # node_feat = torch.cat(node_feat, dim=1)

        node_feat = node_feat.to(self.device)
        batch_list = batch_list.to(self.device)
        graph_repr_feat = self.graph_pool(node_feat, batch_list)
        # pe_repr_feat = self.graph_pool(pe_repr_feat, batch_list)

        # print(node_repr_feat, edge_repr_feat, graph_repr_feat)

        # (num_atoms, embed_dim), (num_bonds, embed_dim), (1, embed_dim)
        # return node_repr_feat, edge_repr_feat, graph_repr_feat, pe_repr_feat
        return graph_repr_feat, (pe_repr_feat, adj_mat_ab, batch_index_ab)

class PredModel(Module):
    def __init__(self, config: dict):
        super(PredModel, self).__init__()
        # self.config = config

        self.embed_dim = config['model']['embed_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.dropout_rate = config['model']['dropout_rate']

        self.lambda_loss = config['pe']['lambda']
        self.alpha_loss = config['pe']['alpha']
        
        self.tasks = config['tasks']
        self.num_tasks = len(self.tasks)

        # self.atom_feat_names = config['atom_feat_names']
        # self.bond_feat_names = config['bond_feat_names']
        # self.bond_float_names = config['bond_float_names']
        # self.bond_angle_float_names = config['bond_angle_float_names']

        # self.graph_pair_embedding = GraphPairEmbedding(self.atom_feat_names, self.bond_feat_names, self.embed_dim)

        # self.mol = mol
        # self.atom_bond_graph = atom_bond_graph
        # self.bond_angle_graph = bond_angle_graph
        self.encoder = GeoGNNModel(config)

        self.tasks_mlp = ModuleList()

        for _ in range(self.num_tasks):
            mlp = Sequential(
                Linear(self.embed_dim, self.hidden_dim, weight_initializer='glorot'),
                ReLU(),
                Dropout(p=self.dropout_rate),
                # Linear(128, 256, weight_initializer='glorot'),
                # GELU(),
                # Dropout(p=self.dropout_rate),
                Linear(self.hidden_dim, self.hidden_dim, weight_initializer='glorot'),
                ReLU(),
                Dropout(p=self.dropout_rate),
                # Linear(256, 128, weight_initializer='glorot'),
                # GELU(),
                # Dropout(p=self.dropout_rate),
                Linear(self.hidden_dim, 1, weight_initializer='glorot')
            )

            self.tasks_mlp.append(mlp)

        # self.loss_batch_train = UncertaintyLoss(self.num_tasks)
        # self.loss_batch_train = DynamicWeightAverageLoss(self.num_tasks)
        
        # self.loss_batch = StdSmoothL1Loss()
        # self.loss_tasks = StdSmoothL1Loss(reduction='none')
        
        self.loss_batch = SmoothL1Loss()
        self.loss_tasks = SmoothL1Loss(reduction='none')
        
        # self.loss_batch = MSELoss()
        # self.loss_tasks = MSELoss(reduction='none')
        self.loss_pe = LaplacianEigenvectorLoss(self.lambda_loss)

    @property
    def device(self):
        return next(self.parameters()).device

    # def forward(self, atom_bond_graph, bond_angle_graph):
    def forward(self, graph_list, y_list):
        # self.data_loader = MoleculeDataLoader(mol_list, y_list, self.config, batch_size=2, num_workers=1, device=self.device)

        # graph_pair_set = self.data_loader.forward()

        batch_size = len(graph_list)

        # loss = 0.0
        
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        for i in range(batch_size):
            # print(i, mol_list[i].GetProp('_Name'))
            atom_bond_graph, bond_angle_graph = graph_list[i]

            atom_bond_graph_list.append(atom_bond_graph)
            bond_angle_graph_list.append(bond_angle_graph)
            # graph_pair = self.graph_pair_embedding(atom_bond_graph, bond_angle_graph)

        # atom_repr, bond_repr, graph_repr = self.encoder(atom_bond_graph, bond_angle_graph)
        graph_repr_feat, pe_repr = self.encoder(atom_bond_graph_list, bond_angle_graph_list)
        pe_repr_feat, adj_mat, batch_index = pe_repr
        
        # y = atom_bond_graph.y
        # y = torch.Tensor([graph_pair_set.y]).to(self.device)
        y = torch.tensor(y_list).to(self.device)
        y = y.squeeze()

        graph_repr_feat = graph_repr_feat.squeeze()

        print(graph_repr_feat, graph_repr_feat.shape)

        # print(graph_repr)
        y_pred = []

        for task_mlp in self.tasks_mlp:
            y_pred.append(task_mlp(graph_repr_feat).squeeze(1))

        # for name, param in self.tasks_mlp.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data, param.data.shape)

        y_pred = torch.stack(y_pred)
        y_pred = y_pred.squeeze()

        # return y_pred

        # mean of (num_nodes, num_tasks) -> 1
        loss_batch_mean = self.loss_batch(y_pred.float(), y.float())

        # (num_nodes, num_tasks)
        loss_tasks = self.loss_tasks(y_pred.float(), y.float())

        loss_pe = self.loss_pe(pe_repr_feat, adj_mat, batch_index)

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
        # loss_tasks_batch = loss_tasks_batch.mean()
        # loss_pe = loss_pe / batch_size

        # loss_batch_train = self.loss_batch_train(loss_tasks)#, iteration)
        
        # sum of (num_tasks, 1) -> 1
        loss_batch_train = torch.sum(loss_tasks)

        print(loss_tasks, loss_batch_train, loss_batch_mean, loss_pe)

        loss_train_with_pe = loss_batch_train + self.alpha_loss * loss_pe

        return loss_train_with_pe, loss_batch_train, loss_batch_mean, loss_tasks






# # ======================================OLD==========================================================================
# class GeoGNNModel(Module):
#     def __init__(self, config={}):
#         super(GeoGNNModel, self).__init__()

#         self.embed_dim = config.get('embed_dim', 32)
#         self.dropout_rate = config.get('dropout_rate', 0.2)
#         self.layer_num = config.get('layer_num', 8)
#         self.readout = config.get('readout', 'mean')

#         self.atom_feat_names = config['atom_feat_names']
#         self.bond_feat_names = config['bond_feat_names']
#         self.bond_float_names = config['bond_float_names']
#         self.bond_angle_float_names = config['bond_angle_float_names']

#         self.graph_pair_embedding = GraphPairEmbedding(self.atom_feat_names, self.bond_feat_names, self.embed_dim)

#         # self.lin_bond_edge = None
#         # self.lin_bond_angle_edge = None

#         # self.bond_embedding_list = ModuleList()
#         # self.bond_length_embedding_list = ModuleList()
#         # self.bond_angle_embedding_list = ModuleList()

#         # self.init_atom_embedding = AtomEmbedding(self.atom_feat_names, self.embed_dim)
#         # self.init_bond_embedding = BondEmbedding(self.bond_feat_names, self.embed_dim)
#         # self.init_bond_length_embedding = BondLengthEmbedding(self.embed_dim)
#         # self.init_bond_angle_embedding = BondAngleEmbedding(self.embed_dim)

#         # self.embed_to_graph = EmbedToGraph()

#         self.atom_bond_block_list = ModuleList()
#         self.bond_angle_block_list = ModuleList()

#         for layer_idx in range(self.layer_num):
#             # self.bond_embedding_list.append(
#             #     BondEmbedding(self.bond_feat_names, self.embed_dim)
#             # )
#             # self.bond_length_embedding_list.append(
#             #     BondLengthEmbedding(self.embed_dim)
#             # )
#             # self.bond_angle_embedding_list.append(
#             #     BondAngleEmbedding(self.embed_dim)
#             # )

#             self.atom_bond_block_list.append(
#                 GeoGNNBlock(self.embed_dim,
#                             self.dropout_rate,
#                             last_act=(layer_idx != self.layer_num - 1))
#             )
#             self.bond_angle_block_list.append(
#                 GeoGNNBlock(self.embed_dim,
#                             self.dropout_rate,
#                             last_act=(layer_idx != self.layer_num - 1))
#             )

#         # self.graph_pool = global_mean_pool()

#         print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
#         print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
#         print('[GeoGNNModel] layer_num:%s' % self.layer_num)
#         print('[GeoGNNModel] readout:%s' % self.readout)
#         print('[GeoGNNModel] atom_names:%s' % str(self.atom_feat_names))
#         print('[GeoGNNModel] bond_names:%s' % str(self.bond_feat_names))
#         print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
#         print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     # def forward(self, graph_pair) -> tuple[Tensor, Tensor, Tensor]:
#     def forward(self, atom_bond_graph, bond_angle_graph) -> tuple[Tensor, Tensor, Tensor]:
#         graph_pair = self.graph_pair_embedding(atom_bond_graph, bond_angle_graph)

#         atom_node_hidden_feat = graph_pair.x_ab
#         bond_edge_hidden_feat = graph_pair.edge_attr_ab
#         bond_edge_index = graph_pair.edge_index_ab

#         bond_node_hidden_feat = graph_pair.x_ba
#         angle_edge_hidden_feat = graph_pair.edge_attr_ba
#         angle_edge_index = graph_pair.edge_index_ba

#         node_hidden_list = []
#         edge_hidden_list = []

#         for layer_idx in range(self.layer_num):
#             # print('Atom Bond Block #{}'.format(layer_idx + 1))
#             atom_node_hidden_feat = self.atom_bond_block_list[layer_idx](
#                 x=atom_node_hidden_feat,
#                 edge_index=bond_edge_index,
#                 edge_attr=bond_edge_hidden_feat
#             )

#             # curr_edge_hidden_feat = atom_bond_graph['edge_attr']
#             # curr_angle_hidden_feat = bond_angle_graph['edge_attr']

#             # print('Bond Angle Block #{}'.format(layer_idx + 1))
#             bond_node_hidden_feat = self.bond_angle_block_list[layer_idx](
#                 x=bond_node_hidden_feat,
#                 edge_index=angle_edge_index,
#                 edge_attr=angle_edge_hidden_feat
#             )

#             node_hidden_list.append(atom_node_hidden_feat)
#             edge_hidden_list.append(bond_node_hidden_feat)
            
#             # print(bond_node_hidden_feat)

#             # bond_edge_hidden_feat = self.lin_bond_edge(bond_node_hidden_feat.T).T

#             # bond_edge_hidden_feat = double_edge_feature(bond_edge_index, bond_node_hidden_feat)
#             # bond_edge_hidden_feat = atom_bond_graph._get_edge_attribute_as_edge_matrix(bond_node_hidden_feat)
#             bond_edge_hidden_feat = GraphUtils.get_edge_attribute_as_edge_matrix(
#                 self.graph_pair_embedding.atom_bond_adj_mat,
#                 self.graph_pair_embedding.atom_bond_idx_map,
#                 bond_node_hidden_feat
#             )

#             # atom_bond_graph.x = node_hidden_feat
#             # atom_bond_graph.edge_attr = double_edge_feature(atom_bond_graph.edge_index, edge_hidden_feat)
#             # bond_angle_graph.x = edge_hidden_feat

#         node_repr_feat = node_hidden_list[-1]
#         edge_repr_feat = edge_hidden_list[-1]

#         # node_feat = []
#         # for hidden_feat in node_hidden_list:
#         #     node_feat.append(global_mean_pool(hidden_feat, None))

#         # graph_repr_feat = torch.cat(node_feat, dim=1)

#         # batch_list = graph_pair.y_batch

#         graph_repr_feat = global_mean_pool(node_repr_feat, None)

#         # print(node_repr_feat, edge_repr_feat, graph_repr_feat)

#         # (num_atoms, embed_dim), (num_bonds, embed_dim), (1, embed_dim)
#         return node_repr_feat, edge_repr_feat, graph_repr_feat
# # ======================================OLD==========================================================================
# # ======================================OLD==========================================================================
# class PredModel(Module):
#     def __init__(self, config: dict):
#         super(PredModel, self).__init__()
#         # self.config = config

#         self.embed_dim = config.get('embed_dim', 32)
#         self.dropout_rate = config.get('dropout_rate', 0.2)

#         # self.atom_feat_names = config['atom_feat_names']
#         # self.bond_feat_names = config['bond_feat_names']
#         # self.bond_float_names = config['bond_float_names']
#         # self.bond_angle_float_names = config['bond_angle_float_names']

#         # self.data_loader = MoleculeDataLoader(mol_list, y_list, config, batch_size=2, num_workers=1, device=device)
#         # self.data_loader = None

#         # self.graph_pair_embedding = GraphPairEmbedding(self.atom_feat_names, self.bond_feat_names, self.embed_dim)

#         # self.mol = mol
#         # self.atom_bond_graph = atom_bond_graph
#         # self.bond_angle_graph = bond_angle_graph
#         self.encoder = GeoGNNModel(config)

#         self.mlp = Sequential(
#             Linear(self.embed_dim, 128, weight_initializer='glorot'),
#             ReLU(),
#             Dropout(p=self.dropout_rate),
#             Linear(128, 128, weight_initializer='glorot'),
#             ReLU(),
#             Dropout(self.dropout_rate),
#             Linear(128, 1)
#         )

#         self.loss = SmoothL1Loss()

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     # def forward(self, atom_bond_graph, bond_angle_graph):
#     def forward(self, mol_list, y_list):
#         # self.data_loader = MoleculeDataLoader(mol_list, y_list, self.config, batch_size=2, num_workers=1, device=self.device)

#         # graph_pair_set = self.data_loader.forward()

#         batch_size = len(mol_list)

#         loss = 0.0

#         for i in range(batch_size):
#             print(i, mol_list[i].GetProp('_Name'))
#             atom_bond_graph = AtomBondGraphFromMol(mol_list[i], y_list[i], self.device)
#             bond_angle_graph = BondAngleGraphFromMol(mol_list[i], y_list[i], self.device)
#             # graph_pair = self.graph_pair_embedding(atom_bond_graph, bond_angle_graph)

#         # atom_repr, bond_repr, graph_repr = self.encoder(atom_bond_graph, bond_angle_graph)
#             atom_repr, bond_repr, graph_repr = self.encoder(atom_bond_graph, bond_angle_graph)
        
#         # y = atom_bond_graph.y
#         # y = torch.Tensor([graph_pair_set.y]).to(self.device)
#             y = torch.tensor([atom_bond_graph.y]).to(self.device)

#         # print(graph_repr, graph_repr.shape)
#             graph_repr = graph_repr.squeeze()

#         # print(graph_repr)
#             y_pred = self.mlp(graph_repr)

#         # return y_pred

#             sub_loss = self.loss(y_pred, y)

#         loss += sub_loss
#         loss = loss / batch_size

        # return loss
# # ======================================OLD==========================================================================