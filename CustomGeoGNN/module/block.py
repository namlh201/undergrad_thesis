from typing import Optional
from torch import Tensor
from torch.nn import Dropout, Module, GELU, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import LayerNorm, GraphNorm

from .conv_layer import GINConv#, GATv2ConvPE as GATv2Conv
from .utils import UnitNorm

class GeoGNNBlock(Module):
    def __init__(self, embed_dim: int, dropout_rate: float, last_act: bool):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GINConv(self.embed_dim)
        # self.gnn = GATv2Conv(self.embed_dim, self.embed_dim, heads=1, edge_dim=self.embed_dim)#, add_self_loops=False)
        # self.gnn = TransformerConv(self.embed_dim, self.embed_dim, edge_dim=self.embed_dim)
        self.norm = LayerNorm(self.embed_dim)
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

class TransformerBlock(Module):
    def __init__(self, embed_dim: int, num_layers: int=6, heads: int=4, dropout_rate: float=0.5):
        super(TransformerBlock, self).__init__()

        encoder_layer = TransformerEncoderLayer(embed_dim, heads, dropout=dropout_rate, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=LayerNorm(embed_dim))

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        x = x + pe
        return self.transformer_encoder(x + pe)