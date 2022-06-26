import copy
import os
import sys
from pytorch_lightning import seed_everything

sys.path.append('/home/xhh/notebooks/GNN/pytorch-template')
sys.path.append('/Users/synapse/Desktop/Repository/pycharm-workspace/pytorch-template')

import torch
import numpy as np
from torch_geometric.loader import ClusterData
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy

from components.backbone import GCN, GraphSAGE, GAT, GIN, PNA
from src.datamodules.components.data import get_data
from src.datamodules.components.loader import SaintRwLoader, NeighborLoader, ClusterLoader, ShadowLoader, to_sparse
from src.utils.index import setup_seed, Dict, pred_fn, loss_fn
from apex import amp


def train(model, optimizer, metric, train_loader, epoch, grad_norm=None):
    model.train()
    metric.reset()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    nodes = []
    total_loss = 0
    for batch in train_loader:
        batch.to(metric.device)
        nodes.append(batch.num_nodes)

        optimizer.zero_grad()
        y_hat = model(batch.x, batch.adj_t, batch.ego_ptr)
        loss = loss_fn(y_hat, batch.y)

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += loss.item() / batch.batch_size
        metric.update(*pred_fn(y_hat, batch.y))
        pbar.update(batch.batch_size)
    pbar.close()

    print(f'Avg Nodes {sum(nodes) / len(nodes):.2f}')

    return total_loss, metric.compute()


@torch.no_grad()
def full_test(model, metric, data):
    model.eval()
    y_hat, ms = model(data.x, data.adj_t), []
    pred, y = pred_fn(y_hat, data.y)
    for _, mask in data('val_mask', 'test_mask'):
        metric.reset()
        metric(pred[mask], y[mask])
        ms.append(metric.compute())
    return ms


@torch.no_grad()
def mini_test(model, metric, *loaders):
    model.eval()
    ms = []
    for loader in loaders:
        metric.reset()
        for data in loader:
            data.to(metric.device)
            y_hat = model(data.x, data.adj_t, data.ego_ptr)
            metric.update(*pred_fn(y_hat, data.y))
        ms.append(metric.compute())
    return ms


def main(hparams):
    seed_everything(hparams.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    metric = F1Score(average='micro') if hparams.metric == 'micro' else Accuracy()
    metric.to(device)

    data, num_features, num_classes, processed_dir = get_data(name=hparams.dataset, split=hparams.split)
    kwargs = {'batch_size': hparams.batch_size, 'shuffle': True, 'num_workers': 10, 'persistent_workers': True}
    hparams.loader = hparams.loader.lower()
    if hparams.loader == 'sage':
        kwargs.update({'num_neighbors': [5] * 2})
        train_loader = NeighborLoader(data, input_nodes=data.train_mask, **kwargs)
        val_loader = NeighborLoader(data, input_nodes=data.val_mask, **kwargs)
        test_loader = NeighborLoader(data, input_nodes=data.test_mask, **kwargs)
    elif hparams.loader == 'cluster':
        cluster_data = ClusterData(data, num_parts=20, save_dir=processed_dir)
        train_loader = ClusterLoader(cluster_data, **kwargs)
    elif hparams.loader == 'saint':
        train_loader = SaintRwLoader(data, walk_length=2, num_steps=5, sample_coverage=100,
                                     save_dir=processed_dir, **kwargs)
    elif hparams.loader == 'shadow':
        kwargs.update({'depth': 2, 'num_neighbors': 4})
        train_loader = ShadowLoader(data, node_idx=data.train_mask, **kwargs)
        val_loader = ShadowLoader(data, node_idx=data.val_mask, **kwargs)
        test_loader = ShadowLoader(data, node_idx=data.test_mask, **kwargs)
    else:
        raise NotImplementedError

    if hparams.loader in ['ego', 'shadow', 'sage']:
        test = lambda _model: mini_test(_model, metric, val_loader, test_loader)
    else:
        data = to_sparse(copy.copy(data)).to(device)
        test = lambda _model: full_test(_model, metric, data)

    model = GraphSAGE(num_features, hparams.hidden_dim, hparams.conv_layers, num_classes,
                      init_layers=hparams.init_layers, dropout=hparams.dropout, dropedge=hparams.dropedge,
                      jk=hparams.jk, residual=hparams.residual).to(device)

    # runs
    runs = hparams.runs
    best_val, best_test = [], []
    for i in range(1, runs + 1):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

        # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        best_val_acc = test_acc = 0
        test_accs = []
        print(f'------------------------{i}------------------------')

        for epoch in range(1, hparams.epoch + 1):
            loss, train_acc = train(model, optimizer, metric, train_loader, epoch, hparams.grad_norm)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {train_acc:.4f}')
            if epoch % hparams.interval == 0:
                val_acc, tmp_test_acc = test(model)
                test_accs.append(round(tmp_test_acc.item(), 3))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                print(f'Epoch: {epoch:02d}, Loss: {loss: .4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

        best_val.append(float(best_val_acc))
        best_test.append(float(test_acc))
        print(test_accs)

    print(f'Valid: {np.mean(best_val):.4f} +- {np.std(best_val):.4f}')
    print(f'Test: {np.mean(best_test):.4f} +- {np.std(best_test):.4f}')


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    params = Dict({
        # data
        'dataset': 'cora',
        'split': 'full',
        'loader': 'sage',
        'batch_size': 64,
        # model
        'hidden_dim': 128,
        'init_layers': 0,
        'conv_layers': 2,
        'dropout': 0.9,
        'dropedge': 0.,
        'jk': None,
        'residual': None,
        # training
        'seed': 123,
        'lr': 0.001,
        'weight_decay': 0,
        'grad_norm': None,
        'runs': 5,
        'epoch': 50,
        'interval': 1,
        'metric': 'acc',  # micro
    })

    main(params)

    print('\n', params)
