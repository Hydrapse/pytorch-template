import copy

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import ClusterData
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy

from components.gnn_backbone import GCN, GraphSAGE, GAT, GIN
from src.datamodules.datasets.data import get_data
from src.datamodules.datasets.loader import SaintRwLoader, NeighborLoader, ClusterLoader, ShadowLoader, to_sparse
from src.utils.index import setup_seed, Dict, pred_fn, loss_fn


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
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) / batch.batch_size
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
    setup_seed(hparams.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metric = F1Score(average='micro') if hparams.metric == 'micro' else Accuracy()
    metric.to(device)

    data, num_features, num_classes, processed_dir = get_data(name=hparams.dataset, split=hparams.split)
    mask = data.train_mask.sum()
    kwargs = {'batch_size': hparams.batch_size, 'shuffle': True, 'num_workers': 0, 'persistent_workers': False}
    if hparams.loader == 'sage':
        train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[10, 10], **kwargs)
    elif hparams.loader == 'cluster':
        cluster_data = ClusterData(data, num_parts=1500, save_dir=processed_dir)
        train_loader = ClusterLoader(cluster_data, **kwargs)
    elif hparams.loader == 'saint':
        train_loader = SaintRwLoader(data, walk_length=2, num_steps=5, sample_coverage=100,
                                     save_dir=processed_dir, **kwargs)
    elif hparams.loader == 'shadow':
        kwargs.update({'depth': 4, 'num_neighbors': 10})
        train_loader = ShadowLoader(data, node_idx=data.train_mask, **kwargs)
        val_loader = ShadowLoader(data, node_idx=data.val_mask, **kwargs)
        test_loader = ShadowLoader(data, node_idx=data.test_mask, **kwargs)
    else:
        raise NotImplementedError

    if hparams.loader in ['ego', 'shadow']:
        test = lambda _model: mini_test(_model, metric, val_loader, test_loader)
    else:
        data = to_sparse(copy.copy(data)).to(device)
        test = lambda _model: full_test(_model, metric, data)

    model = GraphSAGE(num_features, hparams.hidden_dim, hparams.conv_layers, num_classes,
                      init_layers=hparams.init_layers, dropout=hparams.dropout, dropedge=hparams.dropedge,
                      jk=hparams.jk, residual=hparams.residual).to(device)

    # runs
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    best_val_acc = test_acc = 0
    for epoch in range(1, hparams.epoch + 1):
        loss, train_acc = train(model, optimizer, metric, train_loader, epoch, hparams.grad_norm)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {train_acc:.4f}')
        if epoch % hparams.interval == 0:
            val_acc, tmp_test_acc = test(model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss: .4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    params = Dict({
        # data
        'dataset': 'cora',
        'split': 'full',
        'loader': 'sage',
        'batch_size': 10,
        # model
        'hidden_dim': 128,
        'init_layers': 0,
        'conv_layers': 2,
        'dropout': 0.3,
        'dropedge': 0.,
        'jk': 'last',
        'residual': None,
        # training
        'seed': 123,
        'lr': 0.01,
        'weight_decay': 0,
        'grad_norm': None,
        'epoch': 100,
        'interval': 1,
        'metric': 'acc',  # micro
    })

    main(params)

    print('\n', params)
