import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.inits import reset
from torch_scatter import scatter


class MultGroupConv(nn.Module):

    def __init__(self, conv, num_groups):
        super().__init__()

        self.conv = conv
        self.num_groups = num_groups
        self.group_lin = nn.Linear(conv.in_channels, num_groups, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv)
        self.group_lin.reset_parameters()

    def forward(self, x, adj_t, p=None, batch=None, group_ptr=None):
        if self.num_groups == 1:
            if p is not None: x *= p.view(-1, 1)
            return self.conv(x, adj_t)

        s = self.group_lin(x).softmax(dim=-1).t()  # G * N
        expand_x = s.unsqueeze(-1) * x.unsqueeze(0)  # G * N * F

        # 与 adj 对齐: (G * batch_nodes * batch_size) * F
        batch_size = int(batch.unique().numel() / self.num_groups)
        expand_x = torch.cat([expand_x[:, batch == i, :].view(-1, self.in_channels)
                              for i in range(batch_size)])
        if p is not None:
            expand_x *= p.view(-1, 1)

        out = self.conv(expand_x, adj_t)

        return scatter(out, group_ptr, dim=0, reduce='sum')


class EgoGraphPooling(nn.Module):

    def __init__(self, mode, num_groups=1):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['mean', 'max']

        self.num_groups = num_groups

    def forward(self, x_root, xs, p, batch, group_ptr=None):
        if group_ptr is not None and p.size(0) > xs[0].size(0):
            p = scatter(p, group_ptr, reduce='sum')

        for i, x in enumerate(xs):
            x = x * p.view(-1, 1)
            if self.mode == 'mean':
                xs[i] = global_mean_pool(x, batch)
            elif self.mode == 'max':
                xs[i] = global_max_pool(x, batch)

        x = torch.cat([x_root] + xs, dim=-1)

        if self.num_groups > 1:
            x = x.view(self.node_groups, -1, x.size(-1))  # G * batch_size * F
            x = torch.cat([x[i] for i in range(x.size(0))], dim=-1)  # batch_size * (F * G)

        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mode})'
