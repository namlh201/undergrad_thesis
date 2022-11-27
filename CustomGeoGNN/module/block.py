from __future__ import annotations

from math import sqrt
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList, GELU, ReLU, Sequential#, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import LayerNorm, GraphNorm, Linear, global_mean_pool, MLP
from torch_geometric.utils import softmax
from torch_geometric.typing import PairTensor

from .conv_layer import GINConv#, GATv2ConvPE as GATv2Conv
# from .utils import UnitNorm

class GeoGNNBlock(Module):
    def __init__(self, embed_dim: int, dropout_rate: float, last_act: bool):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GINConv(self.embed_dim)
        # self.gnn = GATv2Conv(self.embed_dim, self.embed_dim, heads=1, edge_dim=self.embed_dim)#, add_self_loops=False)
        # self.gnn = TransformerConv(self.embed_dim, self.embed_dim, edge_dim=self.embed_dim)
        self.norm = LayerNorm(self.embed_dim, mode='graph')
        self.graph_norm = GraphNorm(self.embed_dim)

        if last_act:
            self.act = GELU()
        
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, pe: Optional[Tensor]=None) -> Tensor:
        if pe is not None:
            out_node_feat, out_pe = self.gnn(x, edge_index, edge_attr, pe)

            out_pe = self.norm(out_pe)
            out_pe = self.graph_norm(out_pe)

            out_pe = self.dropout(out_pe)
        else:
            out_node_feat = self.gnn(x, edge_index, edge_attr)

        out_node_feat = self.norm(out_node_feat)
        out_node_feat = self.graph_norm(out_node_feat)

        if self.last_act:
            out_node_feat = self.act(out_node_feat)

            if pe is not None:
                out_pe = self.act(out_pe)

        out_node_feat = self.dropout(out_node_feat)

        out_node_feat = out_node_feat + x # residual

        if pe is not None:
            out_pe = out_pe + pe # residual

        if pe is not None:
            return out_node_feat, out_pe
        else:
            return out_node_feat

class TransformerEncoderLayer(Module):
    def __init__(self, embed_dim: int, heads: int, dropout_rate: float, activation: str, dim_feedforward: int=2048):
        super(TransformerEncoderLayer, self).__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dim_feedforward = dim_feedforward

        if self.activation == 'relu':
            self.act_fn = ReLU
        elif self.activation == 'gelu':
            self.act_fn = GELU

        # weights
        self.lin_Q = Linear(embed_dim, embed_dim * heads, weight_initializer='glorot')
        self.lin_K = Linear(embed_dim, embed_dim * heads, weight_initializer='glorot')
        self.lin_V = Linear(embed_dim, embed_dim * heads, weight_initializer='glorot')

        self.lin_z = Linear(embed_dim * heads, embed_dim, weight_initializer='glorot')

        self.ff = Sequential(
            Linear(self.embed_dim, self.dim_feedforward, weight_initializer='glorot'),
            self.act_fn(),
            Dropout(p=self.dropout_rate),
            Linear(self.dim_feedforward, self.dim_feedforward),
            self.act_fn(),
            Dropout(p=self.dropout_rate),
            Linear(self.dim_feedforward, self.embed_dim)
        )

        # self.ff = MLP(in_channels=self.embed_dim,
        #               hidden_channels=self.dim_feedforward,
        #               out_channels=self.embed_dim,
        #               num_layers=3,
        #               act=self.act_fn(),
        #               dropout=self.dropout_rate)

        self.layer_norm = LayerNorm(embed_dim, mode='graph')

    def forward(self, x: Union[Tensor, PairTensor], index: Tensor) -> Tensor:
        H, C = self.heads, self.embed_dim

        if isinstance(x, Tensor):
            x_i = x
            x_j = torch.stack(list(map(lambda j: x[j], index)))
        else:
            x_i, x_j = x

        # computes Q, K, V
        # (num_heads, num_nodes, embed_dim)
        Q = self.lin_Q(x_i).view(H, -1, C)
        K = self.lin_K(x_j).view(H, -1, C)
        V = self.lin_V(x_j).view(H, -1, C)

        # transposes K into shape (num_heads, embed_dim, num_nodes)
        KT = torch.transpose(K, 1, 2)

        # computes self-attention
        # (num_heads, num_nodes, num_nodes)
        attn = (Q @ KT) / sqrt(self.embed_dim)

        # computes attention score
        # (num_heads, num_nodes, embed_dim)
        score = softmax(attn, index, dim=-1) @ V
        # transposes 'score' into shape(num_nodes, num_heads, embed_dim)
        score = torch.transpose(score, 0, 1)
        # concat
        score = score.reshape(-1, H * C)

        z = self.lin_z(score)

        # add + layer norm
        z = z + x_i
        z = self.layer_norm(z)

        # feed-forward
        z = self.ff(z)

        # add + layer norm
        z = z + x_i
        z = self.layer_norm(z)

        return z

class TransformerBlock(Module):
    def __init__(self, embed_dim: int, num_layers: int=2, heads: int=4, dropout_rate: float=0.5):
        super(TransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout_rate

        self.transformer_encoder = ModuleList()

        for _ in range(self.num_layers):
            self.transformer_encoder.append(TransformerEncoderLayer(self.embed_dim, self.heads, self.dropout_rate, activation='relu'))

        # encoder_layer = TransformerEncoderLayer(embed_dim, heads, dropout=dropout_rate, activation='gelu')
        # self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=LayerNorm(embed_dim))

    def forward(self, x: Tensor, pe: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = len(x)

        # x = x + pe

        x_i = torch.stack(list(map(lambda i: x[i], edge_index[0])))
        x_j = torch.stack(list(map(lambda j: x[j], edge_index[1])))

        out = (x_i, x_j)
        index = edge_index[1]

        for i in range(self.num_layers):
            out = self.transformer_encoder[i](out, index)

            out = out + x_i
            x_i = out

        out = global_mean_pool(out, edge_index[0])

        return out
