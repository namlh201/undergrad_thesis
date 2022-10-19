from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from torch.nn import Sequential, GELU, Tanh, Parameter#, ReLU
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn import MessagePassing, Linear, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

class GINConv(MessagePassing):
    def __init__(self, out_channels: int):
        super(GINConv, self).__init__(aggr='add')

        self.mlp = Sequential(
            Linear(out_channels, out_channels * 2, weight_initializer='glorot'),
            GELU(),
            Linear(out_channels * 2, out_channels, weight_initializer='glorot')
        )

        self.mlp_with_pe = Sequential(
            Linear(out_channels * 2, out_channels * 2, weight_initializer='glorot'),
            GELU(),
            Linear(out_channels * 2, out_channels, weight_initializer='glorot')
        )

        self.mlp_pe = Sequential(
            Linear(out_channels, out_channels * 2, weight_initializer='glorot'),
            GELU(),
            Linear(out_channels * 2, out_channels, weight_initializer='glorot'),
            Tanh()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: Tensor, edge_index: Union[SparseTensor, Tensor], edge_attr: Tensor, pe: Optional[Tensor]=None) -> Tensor:
        # (num_nodes, out_channels)
        out_node = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if pe is not None:
            return out_node, self.update_pe(pe, edge_index)
        
        return out_node

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return x_i + x_j + edge_attr

    def update(self, aggr_out: Tensor, pe: Optional[Tensor]=None) -> Tensor:
        if pe is not None:
            aggr_out = torch.cat([aggr_out, pe], dim=1)
            return self.mlp_with_pe(aggr_out)
        
        return self.mlp(aggr_out)

    def update_pe(self, pe: Tensor, edge_index: Union[SparseTensor, Tensor]):
        if isinstance(edge_index, SparseTensor):
            # check if already got self loop
            if edge_index[0][0] == 1:
                edge_index_self_loop = edge_index
            else:
                edge_index_self_loop = edge_index + torch.eye(len(edge_index), dtype=torch.long)
        else:
            num_nodes = len(pe)
            # create adj matrix
            edge_index_self_loop = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).to_dense()
            
            # check if already got self loop
            if edge_index_self_loop[0][0] == 0:
                edge_index_self_loop = edge_index_self_loop + torch.eye(len(edge_index_self_loop), device=self.device, dtype=torch.long)

        return self.mlp_pe(pe + edge_index_self_loop @ pe)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor, edge_attr: Tensor) -> Tensor:
        # bug, must move to cpu
        ret = adj_t.cpu() @ x.cpu() + edge_attr.cpu()
        return ret.to(self.device)

#TODO: Customise GATv2Conv to add positional encoding
class GATv2ConvPE(GATv2Conv):
    '''
        Src: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv
    '''
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super(GATv2ConvPE, self).__init__(
            in_channels,
            out_channels,
            heads,
            concat,
            negative_slope,
            dropout,
            add_self_loops,
            edge_dim,
            fill_value,
            bias,
            share_weights,
            **kwargs
        )

        if isinstance(in_channels, int):
            self.lin_l_pe = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r_pe = self.lin_l_pe
            else:
                self.lin_r_pe = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l_pe = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r_pe = self.lin_l_pe
            else:
                self.lin_r_pe = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att_pe = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias_pe = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias_pe = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_pe', None)

        self._alpha_pe = None

        self.tanh_pe = Tanh()

        self.reset_parameters_pe()

    def reset_parameters_pe(self):
        self.lin_l_pe.reset_parameters()
        self.lin_r_pe.reset_parameters()
        glorot(self.att_pe)
        zeros(self.bias_pe)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, pe: Tensor = None,
                return_attention_weights: bool = None):
        edge_index_x, edge_attr_x = edge_index.clone(), edge_attr.clone()
        out = super(GATv2ConvPE, self).forward(x, edge_index_x, edge_attr_x, return_attention_weights)

        if pe is None:
            return out
        else:
            H, C = self.heads, self.out_channels

            pe_l: OptTensor = None
            pe_r: OptTensor = None
            assert pe is not None, f"pe isn't supposed to be None"
            assert pe.dim() == 2
            pe_l = self.lin_l_pe(pe).view(-1, H, C)
            if self.share_weights:
                pe_r = pe_l
            else:
                pe_r = self.lin_r_pe(pe).view(-1, H, C)

            assert pe_l is not None
            assert pe_r is not None

            # print(edge_index, edge_index.shape)
            # print(torch.max(edge_index))

            edge_index_pe, edge_attr_pe = edge_index.clone(), edge_attr.clone()

            if self.add_self_loops:
                if isinstance(edge_index, Tensor):
                    num_nodes = pe_l.size(0)
                    if pe_r is not None:
                        num_nodes = min(num_nodes, pe_r.size(0))
                    edge_index_pe, edge_attr_pe = remove_self_loops(
                        edge_index_pe, edge_attr_pe)
                    edge_index_pe, edge_attr_pe = add_self_loops(
                        edge_index_pe, edge_attr_pe,
                        fill_value=self.fill_value,
                        num_nodes=num_nodes)
                elif isinstance(edge_index, SparseTensor):
                    if self.edge_dim is None:
                        edge_index = set_diag(edge_index)
                    else:
                        raise NotImplementedError(
                            "The usage of 'edge_attr' and 'add_self_loops' "
                            "simultaneously is currently not yet supported for "
                            "'edge_index' in a 'SparseTensor' form")

            # propagate_type: (x: PairTensor, edge_attr: OptTensor)
            # print(pe, pe.shape)
            # print(pe_l, pe_l.shape)
            # print(pe_r, pe_r.shape)
            # print(edge_index_pe, edge_index_pe.shape)
            # print('pe', pe, pe.shape, pe.size(0))
            # print('edge_index', edge_index, edge_index.shape, edge_index.max())
            # print('edge_index_pe', edge_index_pe, edge_index_pe.shape, edge_index_pe.max())
            # print('pe_l', pe_l, pe_l.shape, pe_l.size(0))
            # print('pe_r', pe_r, pe_r.shape, pe_r.size(0))
            assert edge_index_pe.max() < pe_l.size(0)
            assert edge_index_pe.max() < pe_r.size(0)
            out_pe = self.propagate(edge_index_pe, x=(pe_l, pe_r), edge_attr=edge_attr_pe)

            alpha_pe = self._alpha_pe
            self._alpha_pe = None

            if self.concat:
                out_pe = out_pe.view(-1, self.heads * self.out_channels)
            else:
                out_pe = out_pe.mean(dim=1)

            if self.bias is not None:
                out_pe = out_pe + self.bias

            out_pe = self.tanh_pe(out_pe)

            if isinstance(return_attention_weights, bool):
                assert alpha_pe is not None
                if isinstance(edge_index, Tensor):
                    out_node, (edge_index, alpha) = out
                    return out_node, out_pe, edge_index, alpha, alpha_pe
                elif isinstance(edge_index, SparseTensor):
                    out_node, edge_index = out
                    edge_index_pe = edge_index.clone()
                    return out_node, out_pe, edge_index.set_value(alpha, layout='coo'), edge_index_pe.set_value(alpha_pe, layout='coo')
            else:
                out_node = out
                return out_node, out_pe
