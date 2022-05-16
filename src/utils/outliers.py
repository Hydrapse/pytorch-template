import math

import torch


def replace_nan(val, nan_to=0., inf_to=1.):
    val = torch.where(torch.isinf(val), torch.full_like(val, nan_to), val)
    return torch.where(torch.isnan(val), torch.full_like(val, inf_to), val)


def clip_int(val, lower, upper):
    val = 0 if math.isnan(val) else round(val)
    return max(lower, min(val, upper))
