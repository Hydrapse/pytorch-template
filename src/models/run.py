import sys
import os

curPath = os.path.abspath(os.path.dirname('/home/xhh/notebooks/GNN/pytorch-template/notebooks/'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.models.components.assort_sampler import AdaptiveSampler
from torch_geometric.nn import SAGEConv, global_mean_pool, GCNConv
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, num_features, num_classes, processed_dir = get_data('cora', split='full')

kwargs = {'batch_size': 70,
          # 'num_workers': 0,
          # 'persistent_workers': False,
          # 'pin_memory': True,
          }

sampler = AdaptiveSampler(data, 50, max_hop=10, alpha=0.5)

train_loader = EgoGraphLoader(data.train_mask, sampler, shuffle=True, **kwargs)
val_loader = EgoGraphLoader(data.val_mask, sampler, **kwargs)
test_loader = EgoGraphLoader(data.test_mask, sampler, **kwargs)


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
            x = conv(x, adj_t)
            if i < len(self.convs):
                x = x.relu_()
                x = F.dropout(x, p=0.6, training=self.training)

        # x = x * p
        x = torch.cat([x[root_ptr], global_mean_pool(x * p, batch)], dim=-1)
        x = self.lin(x)

        return x


model = GNN(num_features, 128, num_classes).to(device)

params = list(sampler.parameters()) + list(model.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    nodes, hops = [], []
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        nodes.append(batch.num_nodes)
        hops.append(batch.hop.to(torch.float).mean())

        out = model(batch.x, batch.adj_t, batch.p, batch.batch, batch.ego_ptr)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()

        # print(f'threshold: {sampler.w_threshold.grad}, ego: {sampler.w_ego_u}')
        optimizer.step()
        total_loss += float(loss) * batch.num_graphs
        total_examples += batch.num_graphs
        optimizer.zero_grad()

        correct = int((out.argmax(dim=-1) == batch.y).sum()) / batch.num_graphs
        # print(correct, float(loss))

        pbar.update(batch.batch_size)
    pbar.close()

    train_loss = total_loss / total_examples

    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Avg Nodes {sum(nodes) / len(nodes):.0f}, '
          f'Mean Hop: {sum(hops) / len(hops):.2f}')

    return train_loss


@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = total_examples = 0
    edges = []
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        edges.append(batch.num_edges)

        out = model(batch.x, batch.adj_t, batch.p, batch.batch, batch.ego_ptr)

        correct = int((out.argmax(dim=-1) == batch.y).sum())
        total_correct += correct
        total_examples += batch.num_graphs
        # print(i, correct / batch.num_graphs)

    # print(f'Avg Edges {sum(edges)/len(edges):.2f}')
    return total_correct / total_examples


def main():
    setup_seed(123)

    best_val_acc = test_acc = 0
    for epoch in range(1, 100):
        train_loss = train(epoch)

        if epoch % 1 == 0:
            val_acc = test(val_loader)
            tmp_test_acc = test(test_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:02d}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == '__main__':
    main()
    # print(sampler([26354]))
