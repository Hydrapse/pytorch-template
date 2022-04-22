import copy
import math
from collections import defaultdict
from typing import Optional

import torch
import torch_geometric.transforms
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import Batch, Data

from src.utils.index import clip


class EgoData(Data):
    def __init__(self, x, edge_index, y, p=None):
        super().__init__(x, edge_index, y=y)
        self.p = p


class AdaptiveSampler(nn.Module):

    def __init__(
            self,
            data: Data,
            node_budget: int = 200,
            alpha: float = 0.5,
            p_gather: str = 'sum',  # sum, mean, max, min
            undirected=False,
            threshold_init=1e-6,
            max_hop=10
    ):
        super().__init__()

        self.x = data.x
        self.y = data.y
        self.device = data.x.device
        self.edge_index = data.edge_index
        row, col = self.edge_index.cpu()
        self.adj_t = SparseTensor(
            row=row, col=col,
            value=torch.arange(col.size(0)),  # .to(self.device)
            sparse_sizes=(data.num_nodes, data.num_nodes)).t()

        self.node_budget = node_budget
        self.a = alpha
        self.p_gather = p_gather
        self.undirected = undirected
        self.threshold_init = threshold_init
        self.max_hop = max_hop

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.n_imp = self.heuristic_importance()

        self.w_ego_root = Parameter(torch.Tensor(data.num_features))
        self.w_ego_u = Parameter(torch.Tensor(data.num_features))
        self.w_layer_v = Parameter(torch.Tensor(data.num_features, 1))
        self.w_layer_u = Parameter(torch.Tensor(data.num_features, 1))
        self.w_threshold = Parameter(torch.Tensor(data.num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        w_list = [self.w_ego_u, self.w_ego_root, self.w_layer_u, self.w_layer_v]

        for weight in w_list:
            bound = 1.0 / math.sqrt(weight.size(-1))
            torch.nn.init.uniform_(weight.data, -bound, bound)

        torch.nn.init.zeros_(self.w_threshold)
        # torch.nn.init.uniform_(self.w_threshold, 0, self.threshold_init)

    def forward(self, batch_nodes):
        """"""
        batch_nodes = torch.tensor(batch_nodes)

        thresholds = (self.x[batch_nodes] @ self.w_threshold).view(-1)

        ego_graphs = []
        for i in range(batch_nodes.shape[-1]):
            ego_graphs.append(
                self.sample_receptive_field(batch_nodes[i:i + 1], thresholds[i]))

        batch_data = Batch.from_data_list(ego_graphs)
        batch_data.ego_ptr = (batch_data.ego_ptr + batch_data.ptr[:-1])
        batch_data.batch_size = batch_data.ego_ptr.size(0)

        batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index)
        delattr(batch_data, 'edge_index')
        return batch_data

    def sample_receptive_field(self, batch_node, threshold):
        x = self.x
        e_id = []  # 存储所有边index
        # n_id = [batch_node.to(self.device)]
        n_id = [batch_node]  # 存储所有点index
        # n_p = [torch.tensor([1.]).to(self.device)]
        n_p = [torch.tensor([1.])]  # 储存所有点权重

        p_norm = float('-inf')
        budget = self.node_budget
        hop = 0
        v = batch_node
        while budget > 0 and hop < self.max_hop:
            hop += 1
            adj_t, u = self.adj_t.sample_adj(v, -1, replace=False)
            u_size = u.size(-1)
            row, col, layer_e = adj_t.coo()
            adj = SparseTensor(row=col, col=row, sparse_sizes=adj_t.sparse_sizes()[::-1])

            """计算p_u"""
            ego_score = self.ego_kernel(x[batch_node], x[u])
            layer_score = self.layer_kernel(x[v], x[u], adj)
            p_u = (self.a * ego_score + (1 - self.a) * layer_score) * self.n_imp[u]
            # p_norm = max(torch.max(p_u).detach(), p_norm)
            if hop == 1: p_norm = p_u[0].clone().detach()

            # alpha = 0.01
            # coe = pow(1-alpha, hop-1)  # pagerank
            # coe = self.max_hop - hop / self.max_hop  # markvo
            # coe = budget / self.node_budget  # adapt

            p_u = p_u / (p_norm + 1e-7) + 1.0

            """计算mask"""
            p_clip = torch.clamp(p_u, min=1e-2, max=1)
            num_sample = sum(p_clip).item()
            num_sample = 0 if math.isnan(num_sample) else round(num_sample)
            num_sample = clip(num_sample, 1, budget)
            # print(hop, num_sample)
            mask = torch.zeros((u_size,), dtype=torch.bool)
            mask[torch.multinomial(p_clip, num_sample)] = 1
            budget -= num_sample

            # mask = torch.bernoulli(torch.clamp(p_u, min=0, max=1)).to(torch.bool)
            # layer_cost = sum(mask).item()
            # if layer_cost > budget:
            #     _, p_id = torch.sort(p_u[mask], dim=-1, descending=True)
            #     mask = torch.zeros_like(mask)
            #     mask[p_id[:budget]] = 1
            # budget -= layer_cost

            old_mask = mask
            # p_u -= threshold  # 为了让threshold可导, 自适应 threshold
            mask = mask & (p_u > 0)  # 0 threshold
            if sum(mask).item() < 1:
                if not e_id:
                    mask = old_mask
                    budget = -1
                else:
                    break

            edges = [layer_e[col == i] for i in torch.arange(u_size)[mask]]
            e_id.append(torch.cat(edges))
            n_id.append(u[mask])
            n_p.append(p_u[mask])

            v = u[mask]  # .to(torch.device('cpu'))

        e_id = torch.cat(e_id).unique()
        # print([len(ids) for ids in n_id])
        n_id, n_mask = torch.cat(n_id).unique(return_inverse=True)
        p = scatter(torch.cat(n_p), n_mask, dim=-1, reduce=self.p_gather)

        batch_edge = self.edge_global_to_batch(n_id, e_id)
        ego_data = EgoData(x[n_id], batch_edge, self.y[n_id[n_mask[0]]], p)  # y[batch_node]
        ego_data.hop = hop
        ego_data.ego_ptr = n_mask[0]
        return ego_data

    def ego_kernel(self, h_root, h_u):
        h_root = h_root * self.w_ego_root
        h_u = h_u * self.w_ego_u

        return self.cos(h_root, h_u)

    def layer_kernel(self, h_v, h_u, adj):
        h_v = h_v @ self.w_layer_v
        h_u = h_u @ self.w_layer_u
        h_msg = matmul(adj, h_v, reduce='sum')
        # h_msg = adj @ h_v

        return F.normalize(F.relu(h_msg + h_u), dim=0).view(-1)

    def heuristic_importance(self):
        """这里是全局重要性， 并不只是针对被采样节点的重要性（参考LADIES）"""
        adj_t = self.adj_t.fill_value(1., dtype=torch.float)
        adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        return adj_t.sum(dim=1).pow(0.5)

    def edge_global_to_batch(self, n_id, e_id):
        """ 将全局edgeId转换成batch内id """
        e_idx = self.edge_index[:, e_id]
        batch_idx = []
        for edge in e_idx.t():
            batch_idx.append([
                (n_id == edge[0]).nonzero(as_tuple=True)[0],
                (n_id == edge[1]).nonzero(as_tuple=True)[0]
            ])

        batch_idx = torch.tensor(batch_idx).t()
        batch_idx, _ = add_remaining_self_loops(batch_idx)
        if self.undirected:
            batch_idx = to_undirected(batch_idx)

        val, idx = torch.sort(batch_idx[0])
        return torch.stack([val, batch_idx[1][idx]])

    def get_adj_t(self, num_nodes):
        row, col = self.edge_index.cpu()
        return SparseTensor(
            row=row, col=col, value=torch.arange(col.size(0)),
            sparse_sizes=(num_nodes, num_nodes)).t()
