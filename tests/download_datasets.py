from torch_geometric.datasets import Planetoid, PPI, Flickr, Reddit2, Yelp, WikipediaNetwork, Actor, WebKB, Reddit

# transductive
Planetoid("/mnt/nfs-ssd/raw-datasets/pyg-format/Planetoid", 'Cora')
Planetoid("/mnt/nfs-ssd/raw-datasets/pyg-format/Planetoid", 'CiteSeer')
Planetoid("/mnt/nfs-ssd/raw-datasets/pyg-format/Planetoid", 'PubMed')
# hetero
WikipediaNetwork("/mnt/nfs-ssd/raw-datasets/pyg-format/WikipediaNetwork", 'chameleon')
WikipediaNetwork("/mnt/nfs-ssd/raw-datasets/pyg-format/WikipediaNetwork", 'squirrel')
Actor("/mnt/nfs-ssd/raw-datasets/pyg-format/Actor")
WebKB("/mnt/nfs-ssd/raw-datasets/pyg-format/WebKB", 'Cornell')
WebKB("/mnt/nfs-ssd/raw-datasets/pyg-format/WebKB", 'Texas')
WebKB("/mnt/nfs-ssd/raw-datasets/pyg-format/WebKB", 'Wisconsin')

# inductive
Flickr("/mnt/nfs-ssd/raw-datasets/pyg-format/Flickr")
Reddit("/mnt/nfs-ssd/raw-datasets/pyg-format/Reddit")
Reddit2("/mnt/nfs-ssd/raw-datasets/pyg-format/Reddit2")
# multi-label
PPI("/mnt/nfs-ssd/raw-datasets/pyg-format/PPI")
Yelp("/mnt/nfs-ssd/raw-datasets/pyg-format/Yelp")