from __future__ import annotations

import torch
# from torch import Tensor
# import torch.nn as nn
from torch_geometric.utils import add_self_loops, remove_self_loops, get_self_loop_attr
from torch_sparse import SparseTensor
# from tqdm import tqdm

class GraphUtils():
    @staticmethod
    def _get_matrix_map_node_to_edge_feat(edge_index: torch.Tensor, node_edge_idx_map: dict[tuple, int]) -> torch.Tensor:
        # Get a tensor that map a pair of node indices and edge index to the corresponding edge feature
        num_nodes = int(torch.max(edge_index))
        num_edges = len(edge_index.T)

        row = edge_index[0].tolist()
        col = edge_index[1].tolist()

        map_mat = torch.zeros((num_nodes, num_nodes, num_edges))

        for i in range(num_edges):
            src_node_idx = row[i]
            dst_node_idx = col[i]
            edge_idx = node_edge_idx_map[(src_node_idx, dst_node_idx)]
            map_mat[src_node_idx, dst_node_idx, edge_idx] = 1

        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         if adj_mat[i, j] == 1:
        #             entry = GraphUtils._find_corresponding_entry(node_edge_idx_map, i, j)
        #             edge_idx = entry[-1]
        #             map_mat[i, j, edge_idx] = 1

        return map_mat

    @staticmethod
    def _find_corresponding_entry(node_edge_idx_map: torch.Tensor, i: int, j: int):
        for e in node_edge_idx_map:
            if e[0] == i and e[1] == j:
                return e

    @staticmethod
    def _split_adj_mat(adj_mat) -> torch.Tensor:
        # Split an adj matrix
        # E.g.
        # [[0, 1, 2],                   [[[0, 1, 2], 
        #  [1, 2, 3], --------------->    [0, 0, 0],  
        #  [2, 3, 4]]                     [0, 0, 0]],
        #                                [[0, 0, 0],
        #                                 [1, 2, 3],
        #                                 [0, 0, 0]],
        #                                [[0, 0, 0],
        #                                 [0, 0, 0],
        #                                 [2, 3, 4]]]
        size = len(adj_mat)
        adj_mat_stack = []
        for i in range(size):
            m = torch.zeros((size, size))
            m[i, i] = 1
            adj_mat_stack.append(m)
        adj_mat_stack = torch.stack(adj_mat_stack)

        return adj_mat_stack

    @staticmethod
    def get_edge_attribute_as_node_matrix(edge_index: torch.Tensor, adj_mat: torch.Tensor,
                                          node_edge_idx_map: dict[tuple, int], edge_features: torch.Tensor) -> torch.Tensor:
        # adj_mat_splitted = A
        # map_mat = M
        # edge_features = E
        # Returns A x M x E.sum(dim=0)
        map_mat = GraphUtils._get_matrix_map_node_to_edge_feat(edge_index, node_edge_idx_map)
        adj_mat_splitted = GraphUtils.split_adj_mat(adj_mat)

        map_feat = map_mat @ edge_features

        edge_attr = adj_mat_splitted @ map_feat
        edge_attr = edge_attr.sum(dim=0)

        return edge_attr

    @staticmethod
    def get_edge_attribute_as_edge_matrix(edge_index: torch.Tensor, node_edge_idx_map: dict[tuple, int], edge_features: torch.Tensor) -> torch.Tensor:
        # num_nodes = len(adj_mat)
        num_edges = len(edge_index.T)
        row = edge_index[0].tolist() # src node
        col = edge_index[1].tolist() # dst node

        edge_attr = list(map(
            lambda i: edge_features[node_edge_idx_map[(row[i], col[i])]],
            range(num_edges)
        ))

        # edge_attr = []
        # for i in range(num_edges):
        #     src_node_idx = row[i]
        #     dst_node_idx = col[i]
        #     edge_idx = node_edge_idx_map[(src_node_idx, dst_node_idx)]
        #     edge_attr.append(edge_features[edge_idx])

        return torch.stack(edge_attr)

    # @staticmethod
    # def get_edge_attribute_as_edge_matrix(adj_mat: torch.Tensor, node_edge_idx_map: dict[list, int], edge_features: torch.Tensor) -> torch.Tensor:
    #     num_nodes = len(adj_mat)
    #     # num_edges = self.num_edges

    #     edge_attr = []
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             if adj_mat[i, j] == 1:
    #                 # entry = GraphUtils._find_corresponding_entry(node_edge_idx_map, i, j)
    #                 # print(entry)
    #                 # entry = node_edge_idx_map[(i, j)]
    #                 # edge_idx = entry[-1]
    #                 edge_idx = node_edge_idx_map[(i, j)]
    #                 # print(edge_features)
    #                 edge_attr.append(edge_features[edge_idx])

    #     return torch.stack(edge_attr)

    @staticmethod
    def get_adj_mat_from_edge_index(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
        # print(num_nodes, torch.max(edge_index[0]))
        return SparseTensor.from_edge_index(edge_index=edge_index,
                                            edge_attr=torch.full((edge_index.shape[-1], ), 1),
                                            sparse_sizes=(num_nodes, num_nodes)).to_dense()

    @staticmethod
    def get_random_walk_pe(walk_length: int, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Get random-walk positional encoding
        # Src: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/add_positional_encoding.html#AddRandomWalkPE

        # num_nodes = max(int(edge_index.max()) + 1, len(adj_mat))

        # Adds self loops in order to calculate pe
        edge_index_self_loops, edge_attr_self_loops = remove_self_loops(edge_index)
        edge_index_self_loops, edge_attr_self_loops = add_self_loops(edge_index_self_loops, edge_attr_self_loops, num_nodes=num_nodes)

        adj = SparseTensor.from_edge_index(edge_index=edge_index_self_loops,
                                           edge_attr=edge_attr_self_loops,
                                           sparse_sizes=(num_nodes, num_nodes))

        # if num_nodes == 4:
        #     print(edge_index_self_loops)
        #     print(edge_index)
        #     print(adj)

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        # print('num_nodes', num_nodes)
        # print(row, col, value)
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
        pe = torch.stack(pe_list, dim=-1)

        # print(pe_list[0], len(pe_list[0]))
        # print('pe', pe, pe.shape, pe.size(0))

        return pe