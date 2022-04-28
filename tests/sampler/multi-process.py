from torch.utils.data import DataLoader

from src.models.components.assort_sampler import AdaptiveSampler


def sample_sub_graphs(batch_nodes, thresholds, model_s, q):
    ego_graphs = []
    for i in range(batch_nodes.shape[-1]):
        ego_graphs.append(
            model_s.sample_receptive_field(batch_nodes[i:i+1], thresholds[i:i+1]))

    q.put(ego_graphs)


if __name__ == '__main__':
    from time import time
    from torch_geometric.datasets import Flickr
    import torch.multiprocessing as mp
    import torch
    import torch_geometric.transforms as T


    runs = 5
    num_processes = 10
    batch_size = 64

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    dataset = Flickr("/mnt/nfs-ssd/raw-datasets/pyg-format/Flickr", transform=transform)
    data = dataset[0]
    node_index = data.train_mask.nonzero(as_tuple=False).view(-1)
    loader = DataLoader(node_index.tolist(), shuffle=True, batch_size=batch_size)

    sampler = AdaptiveSampler(data, 100, max_hop=10)

    # kwargs = {'batch_size': 32,
    #           'num_workers': 0,
    #           'persistent_workers': False,
    #           'pin_memory': False,
    #           'shuffle': True}
    # sampler = AdaptiveSampler(data, 100, max_hop=10)
    # ego_loader = EgoGraphLoader(data.train_mask, sampler, **kwargs)


    sampler.share_memory()
    # data.share_memory_()
    t = time()
    for j, n_id in enumerate(loader):
        if j == runs:
            break

        phrase = len(n_id) // num_processes + 1

        thresholds = sampler.get_thresholds(n_id)

        q = mp.Queue()
        processes = []
        for i in range(num_processes):
            print('in')
            divided_id = n_id[i*phrase:(i+1)*phrase]
            divided_trd = thresholds[i*phrase:(i+1)*phrase]
            p = mp.Process(target=sample_sub_graphs,
                           args=(divided_id, divided_trd, sampler, q))
            p.start()
            processes.append(p)
            print('out')
        for p in processes:
            p.join()

        ego_datas = []
        for p in processes:
            ego_datas += q.get()
        print(ego_datas)

        # batch_data = Batch.from_data_list(ego_graphs)
        # batch_data.ego_ptr = (batch_data.ego_ptr + batch_data.ptr[:-1])
        #
        # batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index)
        # delattr(batch_data, 'edge_index')

    print(f'{(time() - t) / runs: .2f}s')