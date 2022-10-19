from __future__ import annotations

from torch import Tensor

from torch_geometric.data import Data

class PairDataBatch(Data):
    def __init__(self, edge_index_ab: Tensor=None, x_ab: Tensor=None, edge_attr_ab: Tensor=None, edge_map_ab: Tensor=None,
                       edge_index_ba: Tensor=None, x_ba: Tensor=None, edge_attr_ba: Tensor=None, edge_map_ba: Tensor=None,
                       y: Tensor=None):
        super(PairData, self).__init__()

        self.edge_index_ab = edge_index_ab
        self.x_ab = x_ab
        self.edge_attr_ab = edge_attr_ab
        self.edge_map_ab = edge_map_ab

        self.edge_index_ba = edge_index_ba
        self.x_ba = x_ba
        self.edge_attr_ba = edge_attr_ba
        self.edge_map_ba = edge_map_ba

        self.y = y

class PairData(Data):
    def __init__(self, edge_index_ab: Tensor=None, x_ab: Tensor=None, edge_attr_ab: Tensor=None, edge_map_ab: Tensor=None,
                       edge_index_ba: Tensor=None, x_ba: Tensor=None, edge_attr_ba: Tensor=None, edge_map_ba: Tensor=None,
                       y: Tensor=None):
        super(PairData, self).__init__()

        self.edge_index_ab = edge_index_ab
        self.x_ab = x_ab
        self.edge_attr_ab = edge_attr_ab
        self.edge_map_ab = edge_map_ab

        self.edge_index_ba = edge_index_ba
        self.x_ba = x_ba
        self.edge_attr_ba = edge_attr_ba
        self.edge_map_ba = edge_map_ba

        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_ab':
            return self.x_ab.size(0)
        if key == 'edge_index_ba':
            return self.x_ba.size(0)
        if key == 'edge_map_ab':
            return self.edge_attr_ab.size(0)
        if key == 'edge_map_ba':
            return self.edge_attr_ba.size(0)
        if key == 'y':
            return self.y.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    #TODO: do some shit with batching