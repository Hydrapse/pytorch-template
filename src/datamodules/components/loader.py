from typing import Optional

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import ClusterLoader as ClusterL, NeighborLoader as NeighborL, \
    GraphSAINTRandomWalkSampler, ShaDowKHopSampler
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor

from src.datamodules.components.sampler import AdaptiveSampler


class SetEgoPtr(BaseTransform):

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, 'ego_ptr'):
            if hasattr(data, 'root_n_id'):  # ego sampling
                data.ego_ptr = data.root_n_id
                delattr(data, 'root_n_id')
            elif hasattr(data, 'batch_size'):  # neighbor sampling
                data.ego_ptr = torch.arange(data.batch_size)
                data.y = data.y[:data.batch_size]
            else:  # batch node sampling
                data.ego_ptr = data.train_mask.nonzero(as_tuple=False).view(-1)
                data.y = data.y[data.train_mask]
            data.batch_size = data.ego_ptr.size(0)
        return data


# PYG Batch Data Transform
base_trans = [T.ToSparseTensor(), SetEgoPtr()]
to_sparse, set_ego = T.ToSparseTensor(), SetEgoPtr()


class NeighborLoader(NeighborL):

    def __init__(self, data, **kwargs):
        super().__init__(data, transform=T.Compose(base_trans), **kwargs)


class ClusterLoader(ClusterL):

    def __collate__(self, batch):
        data = super().__collate__(batch)
        transform = T.Compose(base_trans)
        return transform(data)


class SaintRwLoader(GraphSAINTRandomWalkSampler):

    def __collate__(self, data_list):
        data = super().__collate__(data_list)
        transform = T.Compose(base_trans)
        return transform(data)


class ShadowLoader(ShaDowKHopSampler):

    def __init__(self, data: Data, **kwargs):
        super().__init__(to_sparse(data), **kwargs)

    def __collate__(self, n_id):
        data = super().__collate__(n_id)
        return set_ego(data)


class EgoGraphLoader(DataLoader):

    def __init__(self, node_idx: Optional[Tensor],
                 collator: AdaptiveSampler = None,
                 undirected: bool = False,
                 **kwargs):
        self.collator = collator
        self.to_single_layer = collator.to_single_layer
        self.num_groups = collator.num_groups

        self.undirected = undirected

        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        trans = [T.AddSelfLoops()]
        if undirected:
            trans.append(T.ToUndirected())
        trans.append(T.ToSparseTensor())
        self.transform = T.Compose(trans)

        super().__init__(node_idx.tolist(), collate_fn=self.__collate__, **kwargs)

    def __collate__(self, batch_nodes):
        batch_data = self.collator(batch_nodes)
        batch_data.batch_size = len(batch_nodes)
        delattr(batch_data, 'ptr')

        if self.to_single_layer:
            return self.transform(batch_data)

        edge_index = batch_data.edge_index.unique(dim=-1)

        num_expand = batch_data.num_nodes * self.num_groups
        edge_index = add_self_loops(edge_index, num_nodes=num_expand)[0]

        if self.undirected:
            edge_index = to_undirected(edge_index)

        batch_data.adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                        sparse_sizes=(num_expand, num_expand))
        delattr(batch_data, 'edge_index')
        return batch_data