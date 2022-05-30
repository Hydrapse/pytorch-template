import sys
sys.path.append('/home/xhh/notebooks/GNN/pytorch-template')
sys.path.append('/Users/synapse/Desktop/Repository/pycharm-workspace/pytorch-template')

import numpy as np
from pytorch_lightning import seed_everything
from torch_geometric.datasets import Flickr, Reddit2, Planetoid
import torch.nn.functional as F
import torch
import torch_geometric.transforms as T
import torchmetrics
from src.datamodules.components.data import get_data
from torch_geometric.nn.models import basic_gnn

from components.backbone import GCN, GraphSAGE, GAT, GIN
from src.datamodules.components.loader import to_sparse
from src.utils.index import setup_seed, Dict


def train(model, optimizer, data, grad_norm=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    if grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, metric, data):
    model.eval()
    out, ms = model(data.x, data.adj_t), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        metric.reset()
        pred = out[mask].argmax(dim=-1)
        metric(pred, data.y[mask])
        ms.append(metric.compute())
    return ms


def main(hparams):
    seed_everything(hparams.seed)

    # metric = torchmetrics.F1Score(average='micro')
    metric = torchmetrics.Accuracy()

    data, num_features, num_classes, _ = get_data('cornell', split='full')
    to_sparse(data)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    metric.to(device)
    data.to(device)

    model = GraphSAGE(num_features, hparams.hidden_dim, hparams.conv_layers, num_classes,
                      init_layers=hparams.init_layers, dropout=hparams.dropout, dropedge=hparams.dropedge,
                      jk=hparams.jk, residual=hparams.residual).to(device)

    # per run
    runs = hparams.runs
    best_val, best_test = [], []
    for i in range(1, runs + 1):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

        print(f'------------------------{i}------------------------')
        best_val_acc = test_acc = 0
        for epoch in range(1, hparams.epoch):
            loss = train(model, optimizer, data, hparams.grad_norm)
            train_acc, val_acc, tmp_test_acc = test(model, metric, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss: .4f}, Train: {train_acc:.4f}, '
                  f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

        best_val.append(float(best_val_acc))
        best_test.append(float(test_acc))

    print(f'Valid: {np.mean(best_val):.4f} +- {np.std(best_val):.4f}')
    print(f'Test: {np.mean(best_test):.4f} +- {np.std(best_test):.4f}')


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    params = Dict({
        'hidden_dim': 16,
        'init_layers': 0,
        'conv_layers': 2,
        'dropout': 0.5,
        'dropedge': 0.,
        'jk': None,
        'residual': 'sum',
        'lr': 0.01,
        'weight_decay': 0,
        'grad_norm': None,
        'runs': 5,
        'epoch': 300,
        'seed': 123,
    })

    main(params)

    print(params)
