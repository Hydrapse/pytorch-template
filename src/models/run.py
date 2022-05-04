import copy
import sys
import time
from typing import Union

from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch_sparse import SparseTensor, matmul

sys.path.append('/home/xhh/notebooks/GNN/pytorch-template')
sys.path.append('/Users/synapse/Desktop/Repository/pycharm-workspace/pytorch-template')

from src.models.components.assort_sampler import AdaptiveSampler
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv, GATConv
from torch_geometric.utils import add_remaining_self_loops

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import torch
import torch_geometric.transforms as T

# from src.models.components.gnn_backbone import GCN
from src.datamodules.datasets.data import get_data
from src.datamodules.datasets.loader import EgoGraphLoader
from src.utils.index import setup_seed
import numpy as np


class Conv(SAGEConv):
    def forward(self, x, adj, p=None, size=None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x_l = x if p is None else x * p
            x: OptPairTensor = (x_l, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(adj, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, adj_t, p, batch, root_ptr):
        p = p.reshape(-1, 1)

        # x = torch.cat((x, p.view(-1, 1)), dim=-1)
        for i, conv in enumerate(self.convs):
            # c = torch.zeros_like(x)
            # c[root_ptr] = global_mean_pool(x * p, batch)
            # x = x + c
            x = conv(x, adj_t)
            x = x.relu_()
            x = F.dropout(x, p=0.3, training=self.training)
            if i < len(self.convs) - 1:
                x = x * p
            #     x = torch.cat([x, x * p], dim=-1)

        x = torch.cat([x[root_ptr], global_mean_pool(x * p, batch)], dim=-1)
        x = self.lin(x)

        return x


def train(epoch, loader, model, optimizer, device):
    model.train()

    pbar = tqdm(total=int(len(loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = total_correct = 0
    nodes, hops = [], []
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        nodes.append(batch.num_nodes)
        hops.append(batch.hop.to(torch.float).mean())

        out = model(batch.x, batch.adj_t, batch.p, batch.batch, batch.ego_ptr)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        # temp_w = copy.copy(sampler.w_ego_root[0])
        # temp_m = copy.copy(model.lin.weight)
        # print(model.lin.weight)
        # print(sampler.w_ego_root[0])

        # print(f'layer: {sampler.w_layer_u.grad}, ego: {sampler.w_ego_u}')
        optimizer.step()
        # print(torch.equal(temp_w, sampler.w_ego_root[0]))
        # print(torch.equal(temp_m, model.lin.weight))

        total_loss += float(loss) * batch.num_graphs
        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_examples += batch.num_graphs
        optimizer.zero_grad()

        pbar.update(batch.batch_size)
    pbar.close()

    train_loss = total_loss / total_examples
    train_acc = total_correct / total_examples

    print(f'Epoch: {epoch:02d}, Avg Nodes {sum(nodes) / len(nodes):.0f}, '
          f'Mean Hop: {sum(hops) / len(hops):.2f}')

    return train_loss, train_acc


@torch.no_grad()
def test(loader, model, device):
    model.eval()
    total_correct = total_examples = 0
    nodes, hops = [], []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        nodes.append(batch.num_nodes)
        hops.append(batch.hop.to(torch.float).mean())

        out = model(batch.x, batch.adj_t, batch.p, batch.batch, batch.ego_ptr)

        correct = int((out.argmax(dim=-1) == batch.y).sum())
        total_correct += correct
        total_examples += batch.num_graphs
        # print(i, correct / batch.num_graphs)

    return total_correct / total_examples, hops


def main():
    runs, epochs, seed = 5, 20, 123
    setup_seed(seed)

    torch.autograd.set_detect_anomaly(False)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    data, num_features, num_classes, processed_dir = get_data('cora', split='full')

    kwargs = {'batch_size': 70,
              # 'num_workers': 0,
              # 'persistent_workers': False,
              'pin_memory': False,
              'shuffle': True,
              'undirected': False
              }

    sampler = AdaptiveSampler(data, 30, max_hop=10, alpha=0.01, min_nodes=kwargs['batch_size'],
                              p_gather='mean', num_groups=1, group_type='full', ego_mode=False)

    train_loader = EgoGraphLoader(data.train_mask, sampler, **kwargs)
    val_loader = EgoGraphLoader(data.val_mask, sampler, num_workers=0, persistent_workers=False, **kwargs)
    test_loader = EgoGraphLoader(data.test_mask, sampler, num_workers=0, persistent_workers=False, **kwargs)

    # model = GNN(num_features, 128, num_classes).to(device)
    # params = list(sampler.parameters()) + list(model.parameters())
    # optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=0)

    best_val, best_test = [], []
    for i in range(1, runs + 1):
        sampler.reset_parameters()
        model = GNN(num_features, 128, num_classes).to(device)
        # params = list(sampler.parameters()) + list(model.parameters())
        # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0)
        optimizer = torch.optim.Adam(
            [{'params': sampler.parameters(), 'lr': 0.01, 'weight_decay': 1e-4},
             {'params': model.parameters(), 'lr': 0.001}])

        print(f'------------------------{i}------------------------')
        best_val_acc = best_test_acc = 0
        for epoch in range(1, epochs + 1):
            # sampler.min_nodes = 70
            # sampler.max_hop = 3
            train_loss, train_acc = train(epoch, train_loader, model, optimizer, device)
            # sampler.max_hop = 10
            # sampler.min_nodes = 0
            if epoch % 1 == 0 and epoch > 5:
                start_time = time.perf_counter()
                val_acc, val_hop = test(val_loader, model, device)
                tmp_test_acc, test_hop = test(test_loader, model, device)
                # print(f'Val Hop: {sum(val_hop) / len(val_hop):.2f}, Test Hop: {sum(test_hop) / len(test_hop):.2f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = tmp_test_acc
                print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}, '
                      f'Time:{time.perf_counter() - start_time:.2f}s')

        best_val.append(best_val_acc)
        best_test.append(best_test_acc)

    print(f'Valid: {np.mean(best_val):.4f} +- {np.std(best_val):.4f}')
    print(f'Test: {np.mean(best_test):.4f} +- {np.std(best_test):.4f}')


if __name__ == '__main__':
    main()
    # print(sampler([26354]))
