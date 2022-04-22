import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList

from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.typing import Adj

import torch
from torch_geometric.utils import dropout_adj

from src.utils.index import dropout_edge


class MLP(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_layers: int, dropout: float = 0.):
        super().__init__()

        assert num_layers > 0

        self.dropout = dropout
        self.num_layers = num_layers

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, out_channels))
        for dims in range(num_layers - 1):
            self.lins.append(Linear(out_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for i in range(self.num_layers - 1):
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[i].forward(x)
        return x


class GenGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        conv_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        dropout (float, optional): available for edge_index
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        residual (str): 'sum', 'incep'
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        conv_layers: int,
        out_channels: Optional[int] = None,
        init_layers: int = 1,  # 初始化MLP的层数
        dropout: float = 0.0,
        dropedge = 0,
        act: Union[str, Callable, None] = "relu",
        norm: Optional[torch.nn.Module] = None,

        residual: Optional[str] = None,
        jk: Optional[str] = None,
        act_first: bool = False,
        **kwargs,
    ):
        super().__init__()

        from class_resolver.contrib.torch import activation_resolver

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = conv_layers

        self.dropout = dropout
        self.dropedge = dropedge
        self.act = activation_resolver.make(act)
        assert residual in ['sum', 'incep', None]
        self.residual = residual
        self.jk_mode = jk
        self.act_first = act_first

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if init_layers > 0:
            self.mlp = MLP(in_channels, hidden_channels, init_layers)
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(conv_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if jk is not None:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))

        self.norms = None
        if norm is not None:
            self.norms = ModuleList()
            for _ in range(conv_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm))

        num_layers = conv_layers + (1 if hasattr(self, 'mlp') else 0)
        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)
        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        if hasattr(self, 'mlp'):
            self.mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, ego_ptr = None, batch_idx = None, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []

        x_0 = x_pre = None
        if hasattr(self, 'mlp'):
            x = self.mlp(x)
            x_0 = x_pre = x
            xs.append(x)

        edge_index = dropout_edge(edge_index, p=self.dropedge, training=self.training)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)

            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # TODO: multi-scale update
            if x_0 is not None and x_pre is not None:
                if self.residual == 'sum':
                    x = x + x_pre
                    x_pre = x
                elif self.residual == 'incep':
                    x = x + x_0
            else:
                x_0 = x_pre = x
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        if ego_ptr is not None:
            x = x[ego_ptr]
        return x


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')



class GCN(GenGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)



class GraphSAGE(GenGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)



class GIN(GenGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        from torch_geometric.nn.models.mlp import MLP

        mlp = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        return GINConv(mlp, **kwargs)



class GAT(GenGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)



class PNA(GenGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)