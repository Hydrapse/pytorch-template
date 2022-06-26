import sys
import time

from pytorch_lightning import seed_everything

sys.path.append('/home/xhh/notebooks/GNN/pytorch-template')
sys.path.append('/Users/synapse/Desktop/Repository/pycharm-workspace/pytorch-template')

from torch import Tensor, nn
from torch_geometric.typing import OptPairTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul

from src.models.components.backbone import GraphSAGE, GCN, GAT

from src.datamodules.components.sampler import AdaptiveSampler
from torch_geometric.nn import SAGEConv, global_mean_pool

from tqdm import tqdm

import torch.nn.functional as F
import torch

# from src.models.components.gnn_backbone import GCN
from src.datamodules.components.data import get_data
from src.datamodules.components.loader import EgoGraphLoader
import numpy as np

# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Conv(SAGEConv):
    def __init__(self, in_channels, out_channels, groups, batch_size, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        self.batch_size = batch_size
        self.group_lin = nn.Linear(in_channels, groups, bias=False)

    def forward(self, x, adj, **kwargs) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(adj, x=x[0], **kwargs)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor,
                              p=None, batch=None, group_ptr=None) -> Tensor:

        s = self.group_lin(x).softmax(dim=-1).t()  # G * N
        expand_x = s.unsqueeze(-1) * x.unsqueeze(0)  # G * N * F
        # 与 adj 对齐: (G * batch_nodes * batch_size) * F
        expand_x = torch.cat([expand_x[:, batch == i, :].view(-1, self.in_channels)
                              for i in range(self.batch_size)])
        if p is not None:
            expand_x *= p.view(-1, 1)

        adj_t = adj_t.set_value(None, layout=None)
        expand_x = matmul(adj_t, expand_x, reduce=self.aggr)

        return scatter(expand_x, group_ptr, dim=0, reduce='sum')


class Conv2(SAGEConv):
    def __init__(self, in_channels, out_channels, num_groups, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        self.num_groups = num_groups
        self.group_lin = nn.Linear(in_channels, num_groups, bias=False)

    def forward(self, x, adj, p=None, batch=None, group_ptr=None) -> Tensor:
        if self.num_groups == 1:
            if p is not None: x *= p.view(-1, 1)
            return super().forward(x, adj)

        s = self.group_lin(x).softmax(dim=-1).t()  # G * N
        expand_x = s.unsqueeze(-1) * x.unsqueeze(0)  # G * N * F

        # 与 adj 对齐: (G * batch_nodes * batch_size) * F
        batch_size = int(batch.unique().numel() / self.num_groups)
        expand_x = torch.cat([expand_x[:, batch == i, :].view(-1, self.in_channels)
                              for i in range(batch_size)])
        if p is not None:
            expand_x *= p.view(-1, 1)

        out = super().forward(expand_x, adj)
        out = scatter(out, group_ptr, dim=0, reduce='sum')

        return out


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_groups, as_single_layer: bool):
        super().__init__()

        # 把GNN看做单层时只在最后阶段区分不同的组，否则只在conv阶段区分组
        self.edge_groups = 1 if as_single_layer else num_groups
        self.node_groups = 1 if not as_single_layer else num_groups

        self.convs = torch.nn.ModuleList()
        self.convs.append(Conv2(in_channels, hidden_channels, self.edge_groups))
        self.convs.append(Conv2(hidden_channels, hidden_channels, self.edge_groups))
        # self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(2 * self.node_groups * hidden_channels, out_channels)

    def forward(self, x, adj_t, root_ptr, p, batch, group_ptr):
        # x = torch.cat((x, p.view(-1, 1)), dim=-1)
        for i, conv in enumerate(self.convs):
            # c = torch.zeros_like(x)
            # c[root_ptr] = global_mean_pool(x * p, batch)
            # x = x + c
            x = conv(x, adj_t, p=None if i == 0 else p, batch=batch, group_ptr=group_ptr)
            x = x.relu_()
            x = F.dropout(x, p=0.3, training=self.training)

        if self.edge_groups > 1:
            p = scatter(p, group_ptr, reduce='sum')

        x = torch.cat([x[root_ptr], global_mean_pool(x * p.view(-1, 1), batch)], dim=-1)

        if self.node_groups > 1:
            x = x.view(self.node_groups, -1, x.size(-1))  # G * batch_size * F
            x = torch.cat([x[i] for i in range(x.size(0))], dim=-1)  # batch_size * (F * G)

        return self.lin(x)


def train(epoch, loader, model, optimizer, device):
    model.train()

    pbar = tqdm(total=int(len(loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = total_correct = 0
    nodes, hops, stat_hop = [], [], []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        nodes.append(batch.num_nodes)
        hops.append(batch.hop.to(torch.float).mean())
        stat_hop.append(batch.nmd.tolist())

        # start_time = time.perf_counter()

        out = model(batch.x, batch.adj_t, batch.ego_ptr,
                    p=batch.p, batch=batch.batch, group_ptr=batch.group_ptr)

        # print(f'Forward:{time.perf_counter() - start_time:.3f}s')
        # start_time = time.perf_counter()

        loss = F.cross_entropy(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f'Optimize:{time.perf_counter() - start_time:.2f}s\n')

        total_loss += float(loss) * batch.batch_size
        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_examples += batch.batch_size

        pbar.update(batch.batch_size)
    pbar.close()

    train_loss = total_loss / total_examples
    train_acc = total_correct / total_examples

    mean_hop = (sum(hops) / len(hops)).item()
    stat_hop = torch.tensor(stat_hop).sum(dim=0).tolist()

    print(f'Epoch: {epoch:02d}, Avg Nodes {sum(nodes) / len(nodes):.0f}, '
          f'Mean Hop: {mean_hop:.2f}')

    return train_loss, train_acc, round(mean_hop, 2), stat_hop


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    total_correct = total_examples = 0
    nodes, hops = [], []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        nodes.append(batch.num_nodes)
        hops.append(batch.hop.to(torch.float).mean())

        out = model(batch.x, batch.adj_t, batch.ego_ptr,
                    p=batch.p, batch=batch.batch, group_ptr=batch.group_ptr)

        correct = int((out.argmax(dim=-1) == batch.y).sum())
        total_correct += correct
        total_examples += batch.batch_size

    return total_correct / total_examples, hops


def main():
    runs, epochs, seed = 5, 40, 123
    seed_everything(seed, workers=True)

    # torch.autograd.set_detect_anomaly(False)
    sampler_device = torch.device('cuda:0')
    model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, num_features, num_classes, processed_dir = get_data('cora', split='full')

    kwargs = {'batch_size': 64,
              # 'num_workers': 0,
              # 'persistent_workers': False,
              'shuffle': True,
              'undirected': True,
              }
    batch_size = kwargs['batch_size']

    num_groups = 1
    single_layer = False

    sampler = AdaptiveSampler(data, 50, alpha=0.2, max_hop=10, max_degree=-1, min_nodes=batch_size,
                              p_gather='sum', num_groups=num_groups, group_type='full',
                              ego_mode=False, to_single_layer=single_layer).to(sampler_device)

    model = GraphSAGE(sampler.feature_size, 128, 2, num_classes, init_layers=0, dropout=0.9, residual=None,
                      num_groups=num_groups, subg_pool=[], as_single_layer=single_layer, pool_type='max'
                      ).to(model_device)

    train_loader = EgoGraphLoader(data.train_mask, sampler, **kwargs)
    val_loader = EgoGraphLoader(data.val_mask, sampler, num_workers=15, persistent_workers=True, **kwargs)
    test_loader = EgoGraphLoader(data.test_mask, sampler, num_workers=15, persistent_workers=True, **kwargs)

    best_val, best_test = [], []
    for i in range(1, runs + 1):
        sampler.reset_parameters()
        model.reset_parameters()

        optimizer = torch.optim.Adam(
            [{'params': sampler.parameters(), 'lr': 0.001, 'weight_decay': 1e-5},
             {'params': model.parameters(), 'lr': 0.001}])

        print(f'------------------------{i}------------------------')
        best_val_acc = best_test_acc = 0
        test_accs, hops, stat_hops = [], [], []
        for epoch in range(1, epochs + 1):
            sampler.to(sampler_device)  # 为了使用测试时并行加载数据
            train_loss, train_acc, mean_hop, stat_hop = train(epoch, train_loader, model, optimizer, model_device)
            hops.append(mean_hop)
            stat_hops.append(stat_hop)

            if epoch % 1 == 0 and epoch > 0:
                # sampler.to('cpu')  # 为了使用测试时并行加载数据
                start_time = time.perf_counter()
                val_acc, val_hop = test(val_loader, model, model_device)
                tmp_test_acc, test_hop = test(test_loader, model, model_device)
                test_accs.append(round(tmp_test_acc, 3))
                # print(f'Val Hop: {sum(val_hop) / len(val_hop):.2f}, Test Hop: {sum(test_hop) / len(test_hop):.2f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = tmp_test_acc
                print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Train: {train_acc:.4f}, CurVal: {val_acc:.4f}, '
                      f'Val: {best_val_acc:.4f}, CurTest: {tmp_test_acc:.4f}, Test: {best_test_acc:.4f}, '
                      f'Time:{time.perf_counter() - start_time:.2f}s')
                # sampler.to(sampler_device)

        best_val.append(best_val_acc)
        best_test.append(best_test_acc)
        print(test_accs)
        print(hops)
        print(stat_hops)

    print(f'Valid: {np.mean(best_val):.4f} +- {np.std(best_val):.4f}')
    print(f'Test: {np.mean(best_test):.4f} +- {np.std(best_test):.4f}')


if __name__ == '__main__':
    main()
