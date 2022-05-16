from torch_geometric.datasets import Flickr, Reddit2, Planetoid
import torch.nn.functional as F
import torch
import torch_geometric.transforms as T
import torchmetrics

from components.backbone import GCN, GraphSAGE, GAT, GIN


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
def test(model, metric, *datas):
    model.eval()
    ms = []
    for data in datas:
        metric.reset()
        out = model(data.x, data.adj_t)
        pred = out.argmax(dim=-1)
        metric(pred, data.y)
        ms.append(metric.compute())
    return ms


def main():
    # initialize
    to_sparse = T.ToSparseTensor()
    metric = torchmetrics.F1Score(average='micro')
    # metric = torchmetrics.Accuracy()

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    dataset = Flickr("/mnt/nfs-ssd/raw-components/pyg-format/Flickr", transform=transform)

    # dataset = Reddit2("/mnt/nfs-ssd/raw-components/pyg-format/Reddit2", transform=transform)
    # data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

    # dataset = Planetoid("/mnt/nfs-ssd/raw-components/pyg-format/Planetoid", 'Cora', transform=transform)
    data = dataset[0]

    # hyper params
    grad_norm = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric.to(device)
    data.to(device)
    model = GCN(dataset.num_features, 256, 3, dataset.num_classes, init_layers=0,
                dropout=0.5, dropedge=0.5,
                jk=None, skip_conn=None).to(device)

    train_data = data.subgraph(data.train_mask)
    val_data = data.subgraph(data.val_mask)
    test_data = data.subgraph(data.test_mask)

    to_sparse(train_data)
    to_sparse(val_data)
    to_sparse(test_data)

    # per run
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = test_acc = 0
    for epoch in range(1, 5000):
        loss = train(model, optimizer, train_data, grad_norm)
        # train_acc, val_acc, tmp_test_acc = trans_test(model, data)
        train_acc, val_acc, tmp_test_acc = test(model, metric, train_data, val_data, test_data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss: .4f}, Train: {train_acc:.4f}, '
              f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == '__main__':
    main()
    # print(sampler([26354]))




