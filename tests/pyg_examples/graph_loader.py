import copy
import os.path as osp

import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

from torch_geometric.datasets import Reddit2, Flickr, Reddit
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from src.datamodules.datasets.data import get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, num_features, num_classes, processed_dir = get_data('flickr')

# Already send node features/labels to GPU for faster access during sampling:
# data = dataset[0].to(device, 'x', 'y')

# train_data = data.subgraph(data.train_mask)
# val_data = data.subgraph(data.val_mask)
# test_data = data.subgraph(data.test_mask)

# to_sparse = T.ToSparseTensor()
# to_sparse(train_data)
# to_sparse(val_data)
# to_sparse(test_data)

kwargs = {'batch_size': 512, 'num_workers': 0, 'persistent_workers': False}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

# subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
#                                  num_neighbors=[-1], shuffle=False, **kwargs)
# del subgraph_loader.data.x, subgraph_loader.data.y
# subgraph_loader.data.num_nodes = data.num_nodes
# subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.2, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


model = SAGE(num_features, 128, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
metric = torchmetrics.F1Score(average='micro').to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        # y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        y_hat = model(batch.x, batch.adj_t)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


# @torch.no_grad()
# def test():
#     model.eval()
#     ms = []
#     for data in [val_data, test_data]:
#         to_sparse(data)
#         data.to(device)
#         metric.reset()
#         out = model(data.x, data.adj_t)
#         pred = out.argmax(dim=-1)
#         metric(pred, data.y)
#         ms.append(metric.compute())
#     return ms


@torch.no_grad()
def test():
    model.eval()
    # y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y_hat = model(data.x, data.adj_t).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 101):
    loss, train_acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {train_acc:.4f}')
    val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')