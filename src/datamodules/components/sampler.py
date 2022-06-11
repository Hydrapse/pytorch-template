import copy
import math
import time
from typing import Optional, Union

import torch
from torch import nn
from torch import tensor
from torch.nn import Parameter, init
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, sort_edge_index, subgraph
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from torch_geometric.data import Batch, Data

from src.utils.index import mask_to_index
from src.utils.outliers import clip_int, replace_nan


class EgoData(Data):
    def __init__(self, x, edge_index, p=None, num_groups=1, **kwargs):
        super().__init__(x, edge_index, **kwargs)
        self.p = p
        self.num_groups = num_groups

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes * self.num_groups
        elif key == 'group_ptr':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)


class AdaptiveSampler(nn.Module):

    def __init__(
            self,
            data: Data,
            node_budget: int = 200,
            alpha: float = 0.5,
            p_gather: str = 'sum',  # sum, mean, max, min
            max_hop: int = 10,
            max_degree: int = 32,
            min_nodes: int = 10,  # 最小下一层节点数
            num_groups: int = 2,
            group_type: str = 'full',  # full, split
            ego_mode: bool = False,  # 使用比较慢的ego-wise构造root graph
            to_single_layer: bool = False,  # 是否将下游模型看做单层GNN
            # device: torch.device = 'cpu',
    ):
        super().__init__()

        self.device = 'cpu'
        self.data_device = 'cpu'

        data = copy.copy(data).to(self.data_device)
        self.data = data

        # 用于计算采样概率的特征矩阵，在gpu中计算较快
        self.x = data.x.to(self.device)

        # 拓扑结构，在cpu中计算较快
        self.edge_index = add_remaining_self_loops(data.edge_index)[0]
        row, col = self.edge_index
        self.adj_t = SparseTensor(
            row=row, col=col,
            value=torch.arange(col.size(0)).to(self.data_device),
            sparse_sizes=(data.num_nodes, data.num_nodes)).t()

        # 分组模式
        if ego_mode and not to_single_layer:
            raise NotImplementedError
        if num_groups == 1:
            group_type = 'full'

        self.num_features = data.num_features
        self.node_budget = node_budget
        self.alpha = alpha
        self.p_gather = p_gather
        self.max_hop = max_hop
        self.max_degree = max_degree
        self.min_nodes = min_nodes
        self.num_groups = num_groups
        self.group_type = group_type
        self.ego_mode = ego_mode
        self.to_single_layer = to_single_layer

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
        self.feature_size = max(self.num_features) if to_single_layer else data.num_features

        self.w_ego_root = nn.ParameterList()
        self.w_ego_u = nn.ParameterList()
        self.w_layer_v = nn.ParameterList()
        self.w_layer_u = nn.ParameterList()
        self.w_threshold = nn.ParameterList()
        self.bias = nn.ParameterList()

        for g in range(self.num_groups):
            self.w_ego_root.append(Parameter(torch.empty(self.num_features[g])))
            self.bias.append(Parameter(torch.empty(self.num_features[g])))
            self.w_ego_u.append(Parameter(torch.empty(self.num_features[g])))
            self.w_layer_v.append(Parameter(torch.empty(self.num_features[g], 1)))
            self.w_layer_u.append(Parameter(torch.empty(self.num_features[g], 1)))
            self.w_threshold.append(Parameter(torch.empty(self.num_features[g], 1)))

        self.reset_parameters()

    def to(self, device: Optional[Union[int, str, torch.device]], *args, **kwargs):
        super().to(device, *args, **kwargs)

        self.x = self.x.to(device)
        self.device = device

    def reset_parameters(self):
        for g in range(self.num_groups):
            bound = 1.0 / math.sqrt(self.num_features[g])
            for w in [self.w_ego_u[g], self.w_ego_root[g], self.w_layer_u[g], self.w_layer_v[g]]:
                init.uniform_(w.data, -bound, bound)

            # for w in [self.w_ego_u[g], self.w_ego_root[g]]:
            #     torch.nn.init.uniform_(w.data, -bound, bound)
            # for w in [self.w_layer_u[g], self.w_layer_v[g]]:
            #     w.reset_parameters()

            init.ones_(self.bias[g])
            for w in [self.w_threshold[g], self.bias[g]]:
                init.uniform_(w.data, -bound, bound)
                # torch.nn.init.uniform_(w, 0, 1e-6)

    def forward(self, batch_nodes):
        device = self.data_device
        batch_nodes = tensor(batch_nodes, device=device)
        batch_size = len(batch_nodes)

        batch_list = []
        e_full, n_full, n_p_full = [], [], []
        e_batch_full, n_batch_full = [], []
        e_ptr, n_ptr = [], []
        hop_full = torch.zeros(batch_size, dtype=torch.int, device=device)
        for g in range(self.num_groups):
            x = self.x if self.group_type == 'full' else self.x[:, self.x_ptr[g]:self.x_ptr[g+1]]

            if self.to_single_layer:
                if self.ego_mode:
                    data = self.ego_wise_sampling(batch_nodes, x, g)
                else:
                    data = self.batch_wise_sampling(batch_nodes, x, g)

                pad_size = max(self.num_features) - data.num_features
                x = F.pad(data.x, (0, pad_size)) if pad_size > 0 else data.x
                edge_index = data.edge_index.unique(dim=-1)

                ego_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                ego_mask[data.ego_ptr + data.ptr[:-1]] = 1

                # batch_index需要随g递增， ego_mask不需要
                batch_list.append(Data(x, edge_index, p=data.p, ego_ptr=ego_mask,
                                       batch_index=data.batch, hop=data.hop))
            else:
                e_id, n_id, n_p, e_batch, n_batch, hop, nmd = self.batch_wise_sampling(batch_nodes, x, g)
                e_full.append(e_id)
                n_full.append(n_id)
                n_p_full.append(n_p)
                e_batch_full.append(e_batch)
                n_batch_full.append(n_batch)
                e_ptr.append(torch.full((len(e_id),), g, device=device))
                n_ptr.append(torch.full((len(n_id),), g, device=device))
                hop_full += hop.to(device)

        if self.to_single_layer:
            batch_data = Batch.from_data_list(batch_list)
            batch_data.y = self.data.y[batch_nodes]
            batch_data.hop = batch_data.hop.to(float).view(-1, batch_size).mean(dim=0)

            batch_data.batch = batch_data.batch_index
            batch_data.group_ptr = batch_data.ptr
            delattr(batch_data, 'batch_index')
            return batch_data

        e_full, n_full, n_p_full = torch.cat(e_full), torch.cat(n_full), torch.cat(n_p_full)
        e_batch_full, n_batch_full = torch.cat(e_batch_full), torch.cat(n_batch_full)
        e_ptr, n_ptr = torch.cat(e_ptr), torch.cat(n_ptr)

        # start_time = time.perf_counter()

        expand_num_nodes = 0
        egos, p_idx, expand_n_full = [tensor([])] * batch_size, [tensor([])] * batch_size, [tensor([])] * batch_size
        for bn in range(batch_size):
            bn_mask = n_batch_full == bn
            be_mask = e_batch_full == bn
            n_idx, local_n = n_full[bn_mask].unique(return_inverse=True)
            num_nodes = n_idx.size(0)
            root_ptr = local_n[0]

            p_idx[bn] = mask_to_index(bn_mask)
            expand_n_full[bn] = local_n + n_ptr[bn_mask] * num_nodes + expand_num_nodes  # 全局视角下的节点id
            expand_num_nodes += self.num_groups * num_nodes
            # expand_n = local_n + n_ptr[bn_mask] * num_nodes
            # p = scatter(n_p_full[bn_mask], expand_n, dim_size=self.num_groups * num_nodes, reduce=self.p_gather)

            # TODO: subgraph only for group=1
            # expand_e = subgraph(n_idx, self.edge_index, relabel_nodes=True)[0]

            local_e = self.edge_index[:, e_full[be_mask]].unique(return_inverse=True)[1]
            # local_e = sort_edge_index(local_e, num_nodes=num_nodes)
            expand_e = local_e + e_ptr[be_mask] * num_nodes

            ego_data = EgoData(self.data.x[n_idx], expand_e, num_groups=self.num_groups, ego_ptr=root_ptr)
            ego_data.group_ptr = torch.arange(num_nodes, device=device).expand(self.num_groups, -1).reshape(-1)
            egos[bn] = ego_data

        # print(f'Construct Ego Graph:{time.perf_counter() - start_time:.2f}s')

        p_idx, expand_n_full = torch.cat(p_idx), torch.cat(expand_n_full)
        p = scatter(n_p_full[p_idx], expand_n_full, dim_size=expand_num_nodes, reduce=self.p_gather)

        batch_data = Batch.from_data_list(egos)
        batch_data.p = p
        batch_data.y = self.data.y[batch_nodes]
        batch_data.hop = hop_full / self.num_groups

        batch_data.nmd = F.pad(tensor(nmd), (0, self.max_hop-len(nmd)))  # TODO:

        batch_data.ego_ptr += batch_data.ptr[:-1].to(device)
        delattr(batch_data, 'num_groups')
        return batch_data

    def batch_wise_sampling(self, batch_nodes, x, g):
        num_nodes_dist = []  # TODO： 统计数据

        in0 = in1 = in2 = 0

        device = self.data_device
        batch_size = len(batch_nodes)
        h_roots = x[batch_nodes] * self.w_ego_root[g] + self.bias[g]
        # thresholds = F.relu(x[batch_nodes] @ self.w_threshold[g]).view(-1)

        # 初始化batch图 边、节点、权重; 初始化边、节点对应batch_node
        e_id, n_id, n_p = [tensor([], device=device).to(int)], [batch_nodes], [torch.ones(batch_size, device=device)]
        e_batch, n_batch = [tensor([], device=device).to(int)], [torch.arange(batch_size, device=device)]

        # 开始迭代
        budgets = torch.full((batch_size,), self.node_budget, device=device)
        hop = torch.zeros(batch_size, dtype=torch.int, device=device)
        v, batch_ptr = batch_nodes, n_batch[0]
        while sum(budgets) > 0 and len(v) > 0:
            remain_batch = batch_ptr.unique()
            bs = remain_batch.size(0)
            hop[remain_batch] += 1

            # TODO：  self.max_degree if self.training else -1
            adj_t, u = self.adj_t.sample_adj(v, self.max_degree if self.training else -1, replace=False)
            row, col, layer_e = adj_t.coo()
            col = u[col]
            # row, col, layer_e = self.adj_t[v, :].coo()

            time_start = time.perf_counter()

            idxs, invs, ptrs  = [tensor([])] * bs, [tensor([])] * bs, [tensor([])] * bs
            num_nodes_list, inc = [0] * batch_size, [0] * (bs + 1),
            for i, bn in enumerate(remain_batch):
                # 计算当前bn对于边集[row,col]的"起始下标"与"末尾下标+1"
                bn_mask = batch_ptr == bn
                ptr = bn_mask.nonzero(as_tuple=True)[0]
                start, end = (row == ptr[0]).nonzero()[0], (row == ptr[-1]).nonzero()[-1] + 1

                idx, inv = torch.unique(col[start:end], return_inverse=True)

                idxs[i] = idx
                invs[i] = inv + inc[i]
                ptrs[i] = torch.full((len(idx),), bn, device=device)

                num_nodes = len(idx)
                num_nodes_list[bn] = num_nodes
                inc[i + 1] = inc[i] + num_nodes

            u_idx, edge_inv, batch_ptr = torch.cat(idxs), torch.cat(invs), torch.cat(ptrs)
            num_nodes_dist.append(sum(num_nodes_list))

            in0 += time.perf_counter() - time_start
            time_start = time.perf_counter()

            """计算p_u"""
            # alpha = 0.1
            # coe = pow(1-alpha, hop[pre]-1)  # pagerank
            # coe = self.max_hop - hop / self.max_hop  # markvo
            # coe = budget / self.node_budget  # adapt

            # ego_score = self.batch_kernel(h_roots[batch_ptr], x[u[u_idx]], g)
            # layer_score = self.layer_kernel(x[v], x[u], adj, g, hop[pre])[u_idx]

            h_v = x[v] * self.w_layer_v[g].view(-1)
            h_u = x[u_idx] * self.w_ego_u[g] + scatter(h_v[row], edge_inv.to(self.device), dim=0, reduce='sum')
            # matmul(adj, h_v, reduce='sum')
            num_batch_nodes = tensor(num_nodes_list, device=self.device)[batch_ptr].view(-1, 1)
            p_u = self.alpha * self.cos(h_roots[batch_ptr], h_u / num_batch_nodes)
            # + (torch.rand((batch_ptr.shape[0], ), device=self.device) * 2 - 1) * 1e-3)

            p_u = p_u.to(device)

            in1 += time.perf_counter() - time_start
            time_start = time.perf_counter()

            # ego_score = F.relu(h_roots[batch_ptr] + h_msg).sum(dim=-1)  # * self.n_imp[u[u_idx]]
            # ego_score = self.cos(h_roots[batch_ptr], x[u[u_idx]] * self.w_ego_u[g])

            # layer_score = h_u + matmul(adj, h_v, reduce='sum')
            # layer_score = F.normalize(layer_score, dim=0).view(-1)[u_idx]
            # print(p_u.mean(), p_u.std())

            # p_u = (self.a * ego_score + (1 - self.a) * layer_score) \
            # * self.n_imp[u[u_idx]] * (budgets[batch_ptr] / self.node_budget)

            # if max(hop) == 1:
            #     batch_node_pos = [inc[i] + (u_idx[inc[i]:inc[i + 1]] == i).nonzero().item() for i in remain_batch]
            #     p_norm = p_u[batch_node_pos].clone().detach()
            # p_u = p_u

            # p_u = replace_nan(p_u, nan_to=0., inf_to=1.)  # 目前没有用，防止报错

            # pre_budgets = budgets.clone()
            """计算mask"""
            p_clip = torch.abs(p_u - self.alpha + 1)
            num_samples = p_clip.tolist()
            sample_index = [tensor([])] * bs
            terminate_idx = []  # 开销用完，不会参与下一阶段计算
            for i, bn in enumerate(remain_batch):
                start, end = inc[i], inc[i+1]

                # 计算采样个数
                num_sample = sum(num_samples[start:end])
                if num_sample < 1:  # 权重太小，这层不采样
                    hop[bn] -= 1
                    continue
                num_sample = min(round(num_sample), budgets[bn])

                # 节点采样
                sample_index[i] = torch.multinomial(p_clip[start:end], num_sample).add(inc[i])
                budgets[bn] -= num_sample

                # 如果开销耗尽, 则下一层v中不包含该batch（这一层还是包含的）
                if budgets[bn] <= 0 or hop[bn] >= self.max_hop:
                    terminate_idx.append(torch.arange(start, end))

            mask = torch.zeros((u_idx.size(0),), dtype=torch.bool, device=device, requires_grad=False)
            sample_index = torch.cat(sample_index).to(torch.long)
            mask.index_fill_(0, sample_index, 1)

            in2 += time.perf_counter() - time_start

            """不受budget、hop控制的threshold"""
            # p_u -= threshold  # 为了让threshold可导, 自适应 threshold
            # mask = mask & (p_u > 0)
            p_u = p_u[mask] / hop[batch_ptr][mask] - self.alpha + 1  # / hop[batch_ptr][mask]

            """保存当前层采样节点"""
            e_id.append(layer_e[mask[edge_inv]])
            n_id.append(u_idx[mask])
            n_p.append(p_u)
            e_batch.append(batch_ptr[edge_inv][mask[edge_inv]])
            n_batch.append(batch_ptr[mask])

            """设置下一层节点"""
            if len(terminate_idx) > 0:
                mask = mask.clone()  # 为了不影响反向传播
                mask[torch.cat(terminate_idx)] = 0
            if mask.sum() < self.min_nodes:  # 提前终止
                break

            v, batch_ptr = u_idx[mask], batch_ptr[mask]

        e_id, n_id, n_p = torch.cat(e_id), torch.cat(n_id), torch.cat(n_p)
        e_batch, n_batch = torch.cat(e_batch), torch.cat(n_batch)

        # print(f'batch unique: {in0:.2}s, PU: {in1:.2}s, Sample: {in2:.2}s')

        if not self.to_single_layer:
            return e_id, n_id, n_p, e_batch, n_batch, hop, num_nodes_dist

        egos = []
        x = x.to(self.data_device)
        for bn in range(batch_size):
            bn_mask = n_batch == bn
            n_idx, inv_ptr = n_id[bn_mask].unique(return_inverse=True)
            p = scatter(n_p[bn_mask], inv_ptr, dim=-1, reduce=self.p_gather)

            """inv的unique是一个很优雅的local-e, 而且idx是排序过的unique节点id，和p的unique对应上"""
            sub_edge_index = self.edge_index[:, e_id[e_batch == bn]]
            local_e = sort_edge_index(sub_edge_index.unique(return_inverse=True)[1])

            ego_data = EgoData(x[n_idx], local_e, p, ego_ptr=inv_ptr[0])
            egos.append(ego_data)

        batch_data = Batch.from_data_list(egos)
        batch_data.hop = hop
        return batch_data

    def ego_wise_sampling(self, batch_nodes, x, g):
        """"""
        thresholds = (x[batch_nodes] @ self.w_threshold[g]).view(-1)

        ego_graphs = []
        for i in range(len(batch_nodes)):
            ego_graphs.append(
                self.sample_receptive_field(batch_nodes[i:i + 1], x, g, thresholds[i]))

        return Batch.from_data_list(ego_graphs)

    def sample_receptive_field(self, batch_node, x, g, threshold):
        e_id = []  # 存储所有边index
        # n_id = [batch_node.to(self.device)]
        n_id = [batch_node]  # 存储所有点index
        # n_p = [tensor([1.]).to(self.device)]
        n_p = [tensor([1.])]  # 储存所有点权重

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
            p_u = (self.alpha * ego_score + (1 - self.alpha) * layer_score) * self.n_imp[u] \
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
        ego_data = EgoData(x[n_id], batch_edge, p)
        ego_data.hop = hop
        ego_data.ego_ptr = n_mask[0]
        return ego_data

    def ego_kernel(self, h_root, h_u, g):
        h_root = h_root * self.w_ego_root[g]
        h_u = h_u * self.w_ego_u[g]

        return self.cos(h_root, h_u)

    def batch_kernel(self, h_root, h_u, g):
        h_u = h_u * self.w_ego_u[g]
        # msg = F.normalize(h_root, dim=0).view(-1) * F.normalize(h_u, dim=0).view(-1)
        # print(msg.shape)
        # h_msg = h_root * h_u

        return self.cos(h_root, h_u)

    def layer_kernel(self, h_v, h_u, adj, g):
        h_v = h_v @ self.w_layer_v[g]
        h_u = h_u @ self.w_layer_u[g]
        p = F.relu(h_u + matmul(adj, h_v, reduce='sum'))
        return F.normalize(p, dim=0).view(-1)
        # return p.view(-1)

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

        batch_idx = tensor(batch_idx).t()
        val, idx = torch.sort(batch_idx[0])
        return torch.stack([val, batch_idx[1][idx]])
