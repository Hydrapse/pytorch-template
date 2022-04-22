from typing import Tuple

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI)


def get_planetoid(root: str, name: str, split: str = 'public') -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.NormalizeFeatures(), T.AddSelfLoops()])
    dataset = Planetoid(f'{root}/Planetoid', name, split, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_wikics(root: str) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.AddSelfLoops()])
    dataset = WikiCS(f'{root}/WIKICS', transform=transform)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_yelp(root: str) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.AddSelfLoops()])
    dataset = Yelp(f'{root}/YELP', transform=transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_flickr(root: str) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.AddSelfLoops()])
    dataset = Flickr(f'{root}/Flickr', transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_reddit(root: str) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.AddSelfLoops()])
    dataset = Reddit2(f'{root}/Reddit2', transform=transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_ppi(root: str, split: str = 'train') -> Tuple[Data, int, int, str]:
    pre_transform = T.Compose([T.AddSelfLoops()])
    dataset = PPI(f'{root}/PPI', split=split, pre_transform=pre_transform)
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_sbm(root: str, name: str) -> Tuple[Data, int, int, str]:
    pre_transform = T.Compose([T.AddSelfLoops()])
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=pre_transform)
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


pyg_root = '/mnt/nfs-ssd/raw-datasets/pyg-format'
def get_data(name: str, root: str = pyg_root, **kwargs) -> Tuple[Data, int, int, str]:
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name, **kwargs)
    elif name.lower() == 'wikics':
        return get_wikics(root)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() == 'ppi':
        return get_ppi(root, **kwargs)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() == 'yelp':
        return get_yelp(root)
    else:
        raise NotImplementedError