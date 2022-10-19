import torch
import torch.nn as nn

from tqdm import tqdm

from .defunct import PairData
from .module import RBF
from .utils import GraphUtils, VALUES

class GraphPairEmbeddingBatch(nn.Module):
    def __init__(self, atom_feat_names: list[str], bond_feat_names: list[str],
                 walk_length: int, batch_size: int, embed_dim: int):
        super(GraphPairEmbeddingBatch, self).__init__()

        self.atom_bond_graph_embedding_list = nn.ModuleList()
        self.bond_angle_graph_embedding_list = nn.ModuleList()

        for _ in range(batch_size):
            atom_bond_graph_embedding = AtomBondGraphEmbedding(atom_feat_names, bond_feat_names, walk_length, embed_dim)
            bond_angle_graph_embedding = BondAngleGraphEmbedding(bond_feat_names, embed_dim)

            self.atom_bond_graph_embedding_list.append(atom_bond_graph_embedding)
            self.bond_angle_graph_embedding_list.append(bond_angle_graph_embedding)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, atom_bond_graph_list, bond_angle_graph_list) -> PairData:
        edge_index_ab = []
        x_ab = []
        edge_attr_ab = []
        edge_map_ab = {}
        pe_ab = []

        edge_index_ba = []
        x_ba = []
        edge_attr_ba = []
        edge_map_ba = {}

        batch_list = []
        batch_index_ab = [0] # indices of first row and last row of a batch
        # batch_index_ba = [0] # indices of first row and last row of a batch

        ab_node_inc_val = 0
        ab_edge_inc_val = 0
        ba_node_inc_val = 0
        ba_edge_inc_val = 0
        batch_inc_val = 0
        for i in tqdm(range(len(atom_bond_graph_list)), desc='Embedding'):
            atom_bond_graph = atom_bond_graph_list[i]
            bond_angle_graph = bond_angle_graph_list[i]

            atom_bond_idx_map = atom_bond_graph.node_edge_idx_map
            bond_angle_idx_map = bond_angle_graph.node_edge_idx_map

            atom_node_features, bond_edge_features, bond_edge_index, atom_pe = \
                self.atom_bond_graph_embedding_list[i](atom_bond_graph)

            bond_node_features, angle_edge_features, angle_edge_index = \
                self.bond_angle_graph_embedding_list[i](bond_angle_graph)

            '''
                Append features into a list of tensors for atom_bond_graph
            '''
            num_atom_nodes = len(atom_node_features)
            num_bond_edges = max(list(atom_bond_idx_map.values()))
            # Appends atom_node_features
            # (num_nodes, embed_dim)
            x_ab.append(atom_node_features.T)
            # Appends bond_edge_features
            # (num_edges * 2, embed_dim)
            edge_attr_ab.append(bond_edge_features.T)
            # increment edge_index by node
            # (2, num_edges * 2)
            bond_edge_index = bond_edge_index + ab_node_inc_val
            edge_index_ab.append(bond_edge_index)
            # increment edge_map by node and edge
            # new_atom_bond_idx_map = {}
            for keys in atom_bond_idx_map:
                new_key_i = keys[0] + ab_node_inc_val
                new_key_j = keys[1] + ab_node_inc_val
                edge_map_ab[(new_key_i, new_key_j)] = atom_bond_idx_map[keys] + ab_edge_inc_val
            # Appends atom_pe
            # (num_nodes, embed_dim)
            pe_ab.append(atom_pe.T)
            # update node_inc_val by cur_num_nodes
            ab_node_inc_val += num_atom_nodes
            # update edge_inc_val by cur_num_edges
            ab_edge_inc_val += num_bond_edges
            # Appends ab_node_inc_val into batch_index_ab
            batch_index_ab.append(ab_node_inc_val)

            '''
                Append features into a list of tensors for bond_angle_graph
            '''
            num_bond_nodes = len(bond_node_features)
            num_angle_edges = max(list(bond_angle_idx_map.values()))
            # Appends bond_node_features
            # (num_nodes, embed_dim)
            x_ba.append(bond_node_features.T)
            # Appends angle_edge_features
            # (num_edges * 2, embed_dim)
            edge_attr_ba.append(angle_edge_features.T)
            # increment edge_index by node
            # (2, num_edges)
            angle_edge_index = angle_edge_index + ba_node_inc_val
            edge_index_ba.append(angle_edge_index)
            # increment edge_map by node and edge
            # new_bond_angle_idx_map = {}
            for keys in bond_angle_idx_map:
                new_key_i = keys[0] + ba_node_inc_val
                new_key_j = keys[1] + ba_node_inc_val
                edge_map_ba[(new_key_i, new_key_j)] = bond_angle_idx_map[keys] + ba_edge_inc_val
            # update node_inc_val by cur_num_nodes
            ba_node_inc_val += num_bond_nodes
            # update edge_inc_val by cur_num_edges
            ba_edge_inc_val += num_angle_edges
            # # Appends ba_node_inc_val into batch_index_ba
            # batch_index_ba.append(ba_node_inc_val)

            '''
                Update batch_list
            '''
            batch_list.append(torch.full((1, num_atom_nodes), batch_inc_val))

            batch_inc_val += 1

        '''
            Collating above list into a big tensor for atom_bond_graph
        '''
        # update values
        edge_index_ab = torch.cat(edge_index_ab, dim=1)
        x_ab = torch.cat(x_ab, dim=1).T
        edge_attr_ab = torch.cat(edge_attr_ab, dim=1).T
        pe_ab = torch.cat(pe_ab, dim=1).T

        '''
            Collating above list into a big tensor for bond_angle_graph
        '''
        edge_index_ba = torch.cat(edge_index_ba, dim=1)
        x_ba = torch.cat(x_ba, dim=1).T
        edge_attr_ba = torch.cat(edge_attr_ba, dim=1).T

        batch_list = torch.cat(batch_list, dim=1).squeeze()

        adj_mat_ab = GraphUtils.get_adj_mat_from_edge_index(len(x_ab), edge_index_ab)
        adj_mat_ba = GraphUtils.get_adj_mat_from_edge_index(len(x_ba), edge_index_ba)

        return edge_index_ab, x_ab, edge_attr_ab, edge_map_ab, adj_mat_ab, pe_ab, batch_index_ab, \
               edge_index_ba, x_ba, edge_attr_ba, edge_map_ba, adj_mat_ba, \
               batch_list

class GraphPairEmbedding(nn.Module):
    def __init__(self, atom_feat_names: list[str], bond_feat_names: list[str], embed_dim: int):
        super(GraphPairEmbedding, self).__init__()

        self.atom_bond_graph_embedding = AtomBondGraphEmbedding(atom_feat_names, bond_feat_names, embed_dim)
        self.bond_angle_graph_embedding = BondAngleGraphEmbedding(bond_feat_names, embed_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, atom_bond_graph, bond_angle_graph) -> PairData:
        self.atom_bond_adj_mat = atom_bond_graph.adj_mat
        self.bond_angle_adj_mat = bond_angle_graph.adj_mat

        self.atom_bond_idx_map = atom_bond_graph.node_edge_idx_map
        self.bond_angle_idx_map = bond_angle_graph.node_edge_idx_map

        self.y = atom_bond_graph.y

        atom_node_features, bond_edge_features, bond_edge_index = \
            self.atom_bond_graph_embedding(atom_bond_graph)

        bond_node_features, angle_edge_features, angle_edge_index = \
            self.bond_angle_graph_embedding(bond_angle_graph)

        return PairData(edge_index_ab=bond_edge_index,  x_ab=atom_node_features, edge_attr_ab=bond_edge_features,  edge_map_ab=self.atom_bond_idx_map,
                        edge_index_ba=angle_edge_index, x_ba=bond_node_features, edge_attr_ba=angle_edge_features, edge_map_ba=self.bond_angle_idx_map,
                        y=self.y).to(self.device)

class AtomBondGraphEmbedding(nn.Module):
    def __init__(self, atom_feat_names: list[str], bond_feat_names: list[str],
                 walk_length: int, embed_dim: int):
        super(AtomBondGraphEmbedding, self).__init__()

        # nodes
        self.atom_embedding = AtomEmbedding(atom_feat_names, embed_dim)
        self.atom_mass_embedding = AtomMassEmbedding(embed_dim)

        # positional encoding
        # pe.shape = (num_nodes, embed_dim, walk_length)
        # embed into (num_nodes, embed_dim) with bias
        self.walk_length = walk_length
        # self.pe_embedding = nn.Sequential(
        #     nn.Linear(self.walk_length, 1),
        #     Squeeze(),
        #     nn.Linear(embed_dim, embed_dim)
        # )
        # No need to update the last Linear layer's weight, only need to update bias
        # self.pe_embedding[-1].weight = nn.Parameter(torch.ones((embed_dim, embed_dim), dtype=torch.float), requires_grad=False)

        self.pe_embedding = nn.Linear(self.walk_length, embed_dim)

        # edges
        self.bond_embedding = BondEmbedding(bond_feat_names, embed_dim)
        self.bond_length_embedding = BondLengthEmbedding(embed_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, atom_bond_graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_features = self.atom_embedding(atom_bond_graph.node_features).to(self.device)
        atom_features = atom_features + self.atom_mass_embedding(atom_bond_graph.node_features['mass']).to(self.device)
        bond_features = self.bond_embedding(atom_bond_graph.edge_features).to(self.device)
        bond_features = bond_features + self.bond_length_embedding(atom_bond_graph.edge_features['length']).to(self.device)

        edge_index = atom_bond_graph.edge_index
        
        num_nodes = atom_bond_graph.num_nodes

        bond_features = GraphUtils.get_edge_attribute_as_edge_matrix(
            atom_bond_graph.edge_index,
            atom_bond_graph.node_edge_idx_map,
            bond_features
        )

        atom_pe = GraphUtils.get_random_walk_pe(walk_length=self.walk_length,
                                                edge_index=edge_index,
                                                num_nodes=num_nodes)
        atom_pe = atom_pe.to(self.device)

        atom_pe = self.pe_embedding(atom_pe)
        # print(atom_features.shape, atom_pe.shape)

        return atom_features, bond_features, edge_index, atom_pe

        # return Data(x=atom_features,
        #             edge_index=edge_index,
        #             edge_attr=bond_features,
        #             y=self.y)

class BondAngleGraphEmbedding(nn.Module):
    def __init__(self, bond_feat_names: list[str], embed_dim: int):
        super(BondAngleGraphEmbedding, self).__init__()

        # nodes
        self.bond_embedding = BondEmbedding(bond_feat_names, embed_dim)
        self.bond_length_embedding = BondLengthEmbedding(embed_dim)

        # edges
        self.bond_angle_embedding = BondAngleEmbedding(embed_dim)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, bond_angle_graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bond_features = self.bond_embedding(bond_angle_graph.node_features).to(self.device)
        bond_features = bond_features + self.bond_length_embedding(bond_angle_graph.node_features['length']).to(self.device)
        bond_angle_features = self.bond_angle_embedding(bond_angle_graph.edge_features['angle']).to(self.device)

        edge_index = bond_angle_graph.edge_index

        bond_angle_features = GraphUtils.get_edge_attribute_as_edge_matrix(
            bond_angle_graph.edge_index,
            bond_angle_graph.node_edge_idx_map,
            bond_angle_features
        )

        return bond_features, bond_angle_features, edge_index

        # return Data(x=bond_features,
        #             edge_index=edge_index,
        #             edge_attr=bond_angle_features,
        #             y=self.y)

class AtomEmbedding(nn.Module):
    def __init__(self, feat_names: list[str], embed_dim: int):
        super(AtomEmbedding, self).__init__()

        self.feat_names = feat_names
        self.embed_dim = embed_dim

        self.embed_list = nn.ModuleList()
        for name in self.feat_names:
            feat_size = VALUES.get_atom_feat_size(name)
            self.embed_list.append(nn.Embedding(feat_size, self.embed_dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, atom_features: dict[str, torch.Tensor]) -> torch.Tensor:
        out_embed = 0
        for i, name in enumerate(self.feat_names):
            atom_features[name] = atom_features[name].to(self.device)
            out_embed = out_embed + atom_features[name] @ self.embed_list[i](atom_features[name].int())

        return out_embed.sum(dim=0)

class AtomMassEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(AtomMassEmbedding, self).__init__()

        self.embed_dim = embed_dim

        self.centers = list(torch.arange(0, 20, 1))
        self.gamma = 10

        self.block = nn.Sequential(
            RBF(self.centers, self.gamma),
            nn.Linear(len(self.centers), self.embed_dim).cuda()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, atom_mass_feature: torch.Tensor) -> torch.Tensor:
        out_embed = atom_mass_feature.to(self.device)
        out_embed = self.block(out_embed)

        return out_embed

class BondEmbedding(nn.Module):
    def __init__(self, feat_names: list[str], embed_dim: int):
        super(BondEmbedding, self).__init__()

        self.feat_names = feat_names
        self.embed_dim = embed_dim

        self.embed_list = nn.ModuleList()
        for name in self.feat_names:
            feat_size = VALUES.get_bond_feat_size(name)
            self.embed_list.append(nn.Embedding(feat_size, self.embed_dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, bond_features: dict[str, torch.Tensor]) -> torch.Tensor:
        out_embed = 0
        for i, name in enumerate(self.feat_names):
            bond_features[name] = bond_features[name].to(self.device)
            out_embed = out_embed + bond_features[name] @ self.embed_list[i](bond_features[name].int())

        return out_embed.sum(dim=0)

class BondLengthEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(BondLengthEmbedding, self).__init__()

        self.embed_dim = embed_dim

        self.centers = list(torch.arange(0, 2, 0.1))
        self.gamma = 10

        self.block = nn.Sequential(
            RBF(self.centers, self.gamma),
            nn.Linear(len(self.centers), self.embed_dim).cuda()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, bond_length_feature: torch.Tensor) -> torch.Tensor:
        out_embed = bond_length_feature.to(self.device)
        out_embed = self.block(out_embed)

        return out_embed

class BondAngleEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(BondAngleEmbedding, self).__init__()

        self.embed_dim = embed_dim

        self.centers = list(torch.arange(0, torch.pi, 0.1))
        self.gamma = 10

        self.block = nn.Sequential(
            RBF(self.centers, self.gamma),
            nn.Linear(len(self.centers), self.embed_dim).cuda()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, bond_angle_feature: torch.Tensor) -> torch.Tensor:
        out_embed = bond_angle_feature.to(self.device)
        out_embed = self.block(out_embed)

        return out_embed