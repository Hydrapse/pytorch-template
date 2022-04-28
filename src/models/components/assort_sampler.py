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
from torch_geometric.utils import add_remaining_self_loops, to_undirected, sort_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import Batch, Data

from src.utils.index import clip_int, replace_nan


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
            max_hop: int = 10,
            num_groups: int = 2,
            group_type: str = 'full',  # full, split
            ego_mode: bool = False  # 使用比较慢的ego-wise构造rootgraph
    ):
        super().__init__()

        self.x = data.x
        self.y = data.y
        self.device = data.x.device
        self.num_features = data.num_features
        self.edge_index = add_remaining_self_loops(data.edge_index)[0]
        row, col = self.edge_index.cpu()
        self.adj_t = SparseTensor(
            row=row, col=col,
            value=torch.arange(col.size(0)),  # .to(self.device)
            sparse_sizes=(data.num_nodes, data.num_nodes)).t()

        self.node_budget = node_budget
        self.a = alpha
        self.p_gather = p_gather
        self.max_hop = max_hop
        self.num_groups = num_groups
        self.group_type = group_type
        self.ego_mode = ego_mode

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.n_imp = self.heuristic_importance()

        if group_type == 'full':
            self.num_features = [data.num_features] * num_groups
        else:
            group_size = int(data.num_features / num_groups)
            self.x_ptr = list(range(0, data.num_features, group_size))
            self.x_ptr[-1] = data.num_features
            self.num_features = [group_size] * num_groups
            self.num_features[-1] += data.num_features % num_groups

        self.w_ego_root = []
        self.w_ego_u = []
        self.w_layer_v = []
        self.w_layer_u = []
        self.w_threshold = []

        for g in range(self.num_groups):
            self.w_ego_root.append(Parameter(torch.Tensor(self.num_features[g])))
            self.w_ego_u.append(Parameter(torch.Tensor(self.num_features[g])))
            self.w_layer_v.append(Parameter(torch.Tensor(self.num_features[g], 1)))
            self.w_layer_u.append(Parameter(torch.Tensor(self.num_features[g], 1)))
            self.w_threshold.append(Parameter(torch.Tensor(self.num_features[g], 1)))

        self.reset_parameters()

    def reset_parameters(self):
        for g in range(self.num_groups):
            bound = 1.0 / math.sqrt(self.num_features[g])
            # for w in [self.w_ego_u[g], self.w_ego_root[g], self.w_layer_u[g], self.w_layer_v[g]]:
            #     torch.nn.init.uniform_(w.data, -bound, bound)
            for w in [self.w_ego_u[g], self.w_ego_root[g]]:
                torch.nn.init.uniform_(w.data, -bound, bound)
            for w in [self.w_layer_u[g], self.w_layer_v[g]]:
                torch.nn.init.uniform_(w.data, -bound, bound)
            for w in [self.w_threshold[g]]:
                torch.nn.init.zeros_(w)
                # torch.nn.init.uniform_(w, 0, 1e-6)

    def forward(self, batch_nodes):
        batch_nodes = torch.tensor(batch_nodes)

        batch_datas = []
        for g in range(self.num_groups):
            x = self.x if self.group_type == 'full' else self.x[:, self.x_ptr[g]:self.x_ptr[g+1]]
            if self.ego_mode:
                batch_data = self.ego_wise_sampling(batch_nodes, x, g)
            else:
                batch_data = self.batch_wise_sampling(batch_nodes, x, g)
            batch_datas.append(batch_data)

        return batch_datas[0]  # TODO: 为了测试方便, 只取group0

    def batch_wise_sampling(self, batch_nodes, x, g):
        batch_size = len(batch_nodes)
        h_roots = x[batch_nodes] * self.w_ego_root[g]
        # thresholds = (x[batch_nodes] @ self.w_threshold[g]).view(-1)

        # 初始化batch图 边、节点、权重; 初始化边、节点对应batch_node
        e_id, n_id, n_p = [], [batch_nodes], [torch.ones(batch_size)]
        e_batch, n_batch = [], [torch.arange(batch_size)]

        # 开始迭代
        budgets = torch.full((batch_size,), self.node_budget)
        hop, p_norm = torch.zeros(batch_size).to(int), float('-inf')
        v, batch_ptr = batch_nodes, n_batch[0]
        while sum(budgets) > 0 and len(v) > 0:
            remain_batch = batch_ptr.unique().tolist()
            hop[remain_batch] += 1

            adj_t, u = self.adj_t.sample_adj(v, -1, replace=False)
            row, col, layer_e = adj_t.coo()
            adj = SparseTensor(row=col, col=row, sparse_sizes=adj_t.sparse_sizes()[::-1])

            num_nodes_list = torch.zeros_like(batch_nodes)
            idxs, invs, ptrs, inc = [], [], [], [0]
            row = torch.cat([row, torch.Tensor([len(v)]).to(int)])  # 多拼接一位方便索引
            for bn in remain_batch:
                bn_mask = batch_ptr == bn
                ptr = bn_mask.nonzero(as_tuple=True)[0]
                start, end = (row == ptr[0]).nonzero()[0], (row == ptr[-1] + 1).nonzero()[0]
                idx, inv = torch.unique(col[start:end], return_inverse=True)
                idxs.append(idx)
                invs.append(inv + inc[-1])
                ptrs.append(torch.full((len(idx),), bn))
                inc.append(inc[-1] + len(idx))
                num_nodes_list[bn] = len(idx)
            u_idx, edge_inv, batch_ptr = torch.cat(idxs), torch.cat(invs), torch.cat(ptrs)

            """计算p_u"""
            # alpha = 0.01
            # coe = pow(1-alpha, hop-1)  # pagerank
            # coe = self.max_hop - hop / self.max_hop  # markvo
            # coe = budget / self.node_budget  # adapt
            ego_score = self.cos(h_roots[batch_ptr], x[u[u_idx]] * self.w_ego_u[g])
            layer_score = self.layer_kernel(x[v], x[u], adj, g)[u_idx]
            p_u = (self.a * ego_score + (1 - self.a) * layer_score) * self.n_imp[u[u_idx]] \
                  * (budgets[batch_ptr] / self.node_budget)
            # print('\nHop:', hop)
            # print(f'Ego: {ego_score.mean():.4f}, Layer: {layer_score.mean():.4f}, Pre P: {p_u.mean():.4f}', end=', ')
            if max(hop) == 1:
                batch_node_pos = [inc[i] + (u_idx[inc[i]:inc[i + 1]] == i).nonzero().item() for i in remain_batch]
                p_norm = p_u[batch_node_pos].clone().detach()

            p_u = p_u / p_norm[batch_ptr] + 1.0
            p_u = replace_nan(p_u, nan_to=0., inf_to=1.)
            # print(f'Mid mid P:{p_u.mean():.4f}', end=', ')
            # pre_budgets = budgets.clone()

            """计算mask"""
            p_clip = torch.clamp(p_u, min=1e-5, max=1)
            mask = torch.zeros((u_idx.size(0),), dtype=torch.bool)
            terminate_idx = []  # 开销用完，不会参与下一阶段计算
            for i, bn in enumerate(remain_batch):
                start, end = inc[i], inc[i + 1]
                batch_p = p_clip[start:end]
                num_sample = sum(batch_p).item()

                # 默认保底采样
                # num_sample = clip_int(num_sample, 1, budgets[bn])

                # 权重太小，这层不采样
                if num_sample < 1:
                    hop[bn] -= 1
                    continue
                num_sample = min(round(num_sample), budgets[bn])

                mask[start:end][torch.multinomial(batch_p, num_sample)] = 1
                budgets[bn] -= num_sample

                # 如果开销耗尽, 则下一层v中不包含该batch（这一层还是包含的）
                if budgets[bn] <= 0 or hop[bn] >= self.max_hop:
                    terminate_idx.append(torch.arange(start, end))

            # print(f'Post P:{p_u.mean():.4f}')
            # print(f'Full: {num_nodes_list.tolist()} \nSamp: {(pre_budgets-budgets).tolist()}')
            # print(f'P Norm: {p_norm.tolist()}')

            """不受budget、hop控制的threshold"""
            old_mask = mask
            # p_u -= threshold  # 为了让threshold可导, 自适应 threshold
            mask = mask & (p_u > 0)  # Todo: 0 threshold
            if sum(mask).item() < 1:  # TODO: 不够优雅
                if not e_id:
                    mask = old_mask
                    budgets = [-1]
                else:
                    break

            """保存当前层采样节点"""
            e_id.append(layer_e[mask[edge_inv]])
            n_id.append(u[u_idx][mask])
            n_p.append(p_u[mask])
            e_batch.append(batch_ptr[edge_inv][mask[edge_inv]])
            n_batch.append(batch_ptr[mask])

            """设置下一层节点"""
            if len(terminate_idx) > 0:
                mask = mask.clone()  # 为了不影响反向传播
                mask[torch.cat(terminate_idx)] = 0
            v, batch_ptr = u[u_idx][mask], batch_ptr[mask]

        e_id, n_id, n_p = torch.cat(e_id), torch.cat(n_id), torch.cat(n_p)
        e_batch, n_batch = torch.cat(e_batch), torch.cat(n_batch)

        egos = []
        for bn in range(batch_size):
            bn_mask = n_batch == bn
            n_idx, inv_ptr = n_id[bn_mask].unique(return_inverse=True)
            p = scatter(n_p[bn_mask], inv_ptr, dim=-1, reduce='sum')

            """inv的unique是一个很优雅的local-e, 而且idx是排序过的unique节点id，和p的unique对应上"""
            sub_edge_index = self.edge_index[:, e_id[e_batch == bn]]
            local_e = sort_edge_index(sub_edge_index.unique(return_inverse=True)[1])

            ego_data = EgoData(x[n_idx], local_e, self.y[n_idx[inv_ptr[0]]], p)
            ego_data.ego_ptr = inv_ptr[0]
            egos.append(ego_data)

        batch_data = Batch.from_data_list(egos)
        batch_data.ego_ptr = (batch_data.ego_ptr + batch_data.ptr[:-1])
        batch_data.hop = hop
        return batch_data

    def ego_wise_sampling(self, batch_nodes, x, g):
        """"""
        thresholds = (x[batch_nodes] @ self.w_threshold[g]).view(-1)

        ego_graphs = []
        for i in range(len(batch_nodes)):
            ego_graphs.append(
                self.sample_receptive_field(batch_nodes[i:i + 1], x, g, thresholds[i]))

        batch_data = Batch.from_data_list(ego_graphs)
        batch_data.ego_ptr = (batch_data.ego_ptr + batch_data.ptr[:-1])
        return batch_data

    def sample_receptive_field(self, batch_node, x, g, threshold):
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
            ego_score = self.ego_kernel(x[batch_node], x[u], g)
            layer_score = self.layer_kernel(x[v], x[u], adj, g)
            # TODO: 不加拓扑权重反而高一些（full）
            p_u = (self.a * ego_score + (1 - self.a) * layer_score) * self.n_imp[u] \
                  * (budget / self.node_budget)
            # print('\nHop:', hop)
            # print(f'Ego: {ego_score.mean():.4f}, Layer: {layer_score.mean():.4f}, Pre P: {p_u.mean():.4f}', end=', ')
            if hop == 1: p_norm = p_u[0].clone().detach()

            # alpha = 0.01
            # coe = pow(1-alpha, hop-1)  # pagerank
            # coe = self.max_hop - hop / self.max_hop  # markvo
            # coe = budget / self.node_budget  # adapt

            p_u = p_u / p_norm + 1.0
            p_u = replace_nan(p_u, nan_to=0., inf_to=1.)


            """计算mask"""
            p_clip = torch.clamp(p_u, min=1e-5, max=1)
            num_sample = clip_int(sum(p_clip).item(), 1, budget)
            # print(f'Post P:{p_u.mean():.4f}')
            # print(f'Num Nodes: {u_size} \nNum Sample: {num_sample}')
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
            mask = mask & (p_u > 0)  # Todo: 0 threshold
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
        n_id, n_mask = torch.cat(n_id).unique(return_inverse=True)
        p = scatter(torch.cat(n_p), n_mask, dim=-1, reduce=self.p_gather)

        batch_edge = self.edge_global_to_batch(n_id, e_id)
        ego_data = EgoData(x[n_id], batch_edge, self.y[n_id[n_mask[0]]], p)  # y[batch_node]
        ego_data.hop = hop
        ego_data.ego_ptr = n_mask[0]
        return ego_data

    def ego_kernel(self, h_root, h_u, g):
        h_root = h_root * self.w_ego_root[g]
        h_u = h_u * self.w_ego_u[g]

        return self.cos(h_root, h_u)

    def layer_kernel(self, h_v, h_u, adj, g):
        h_v = h_v @ self.w_layer_v[g]
        h_u = h_u @ self.w_layer_u[g]
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
        val, idx = torch.sort(batch_idx[0])
        return torch.stack([val, batch_idx[1][idx]])
