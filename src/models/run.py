import sys
import time

sys.path.append('/home/xhh/notebooks/GNN/pytorch-template')

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

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data, num_features, num_classes, processed_dir = get_data('cora', split='full')

kwargs = {'batch_size': 70,
          # 'num_workers': 0,
          # 'persistent_workers': False,
          'pin_memory': False,
          'shuffle': True,
          'undirected': False
          }

sampler = AdaptiveSampler(data, 50, max_hop=10, alpha=0.7, p_gather='mean',
                          num_groups=1, group_type='full', ego_mode=True)

train_loader = EgoGraphLoader(data.train_mask, sampler, **kwargs)
val_loader = EgoGraphLoader(data.val_mask, sampler, num_workers=0, persistent_workers=False, **kwargs)
test_loader = EgoGraphLoader(data.test_mask, sampler, num_workers=0, persistent_workers=False, **kwargs)


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
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.3, training=self.training)

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

        # print(f'layer: {sampler.w_layer_u.grad}, ego: {sampler.w_ego_u}')
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
    sampler.eval()
    model.eval()
    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)

        out = model(batch.x, batch.adj_t, batch.p, batch.batch, batch.ego_ptr)

        correct = int((out.argmax(dim=-1) == batch.y).sum())
        total_correct += correct
        total_examples += batch.num_graphs
        # print(i, correct / batch.num_graphs)

    return total_correct / total_examples


def main():
    setup_seed(123)

    best_val_acc = test_acc = 0
    for epoch in range(1, 100):
        train_loss = train(epoch)

        if epoch % 1 == 0:
            start_time = time.perf_counter()
            val_acc = test(val_loader)
            tmp_test_acc = test(test_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:02d}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}, '
                  f'Time:{time.perf_counter() - start_time:.2f}s')


if __name__ == '__main__':
    main()
    # print(sampler([26354]))
