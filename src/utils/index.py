import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import dropout_adj
from torch_sparse import SparseTensor


class Dict(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def dropout_edge(edge_index: Adj, p: float, training: bool = True):
    if not training or p == 0.:
        return edge_index

    if isinstance(edge_index, SparseTensor):
        if edge_index.storage.value() is not None:
            value = F.dropout(edge_index.storage.value(), p=p)
            edge_index = edge_index.set_value(value, layout='coo')
        else:
            mask = torch.rand(edge_index.nnz(), device=edge_index.storage.row().device) > p
            edge_index = edge_index.masked_select_nnz(mask, layout='coo')
    else:
        edge_index, edge_attr = dropout_adj(edge_index, p=p)

    return edge_index


def replace_nan(val, nan_to=0., inf_to=1.):
    val = torch.where(torch.isinf(val), torch.full_like(val, nan_to), val)
    return torch.where(torch.isnan(val), torch.full_like(val, inf_to), val)


def clip_int(val, lower, upper):
    val = 0 if math.isnan(val) else round(val)
    return max(lower, min(val, upper))


def setup_seed(seed):
    if seed == -1:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pred_fn(y_hat, y) -> Tuple[Tensor, Tensor]:
    if y.dim() == 1:  # multi-class
        pred = y_hat.argmax(dim=-1)
    else:             # multi-label
        pred = (y_hat > 0).float()
    return pred, y


def loss_fn(y_hat, y) -> Tensor:
    if y.dim() == 1:
        return F.cross_entropy(y_hat, y)
    else:
        return F.binary_cross_entropy_with_logits(y_hat, y)