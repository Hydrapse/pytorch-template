from typing import Any, List, Optional, Union, Callable

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.backbone import GCN, GraphSAGE, GAT, GIN, PNA
from src.utils.index import loss_fn, pred_fn


class GNNModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: str,
        metric: str,
        lr: float,
        weight_decay: float,
        sampler,
        **model_kwargs
    ):
        super().__init__()

        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if sampler is not None:
            self.sampler = sampler

        model = model.lower()
        if model == 'gcn':
            self.model = GCN(**model_kwargs)
        elif model == 'sage':
            self.model = GraphSAGE(**model_kwargs)
        elif model == 'gat':
            self.model = GAT(**model_kwargs)
        elif model == 'gin':
            self.model = GIN(**model_kwargs)
        elif model == 'pna':
            self.model = PNA(**model_kwargs)
        else:
            raise NotImplementedError

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        Metric = Accuracy if metric.lower() == 'acc' else F1Score
        self.train_acc = Metric()
        self.val_acc = Metric()
        self.test_acc = Metric()
        self.val_acc_best = MaxMetric()

    def forward(self, x, adj_t, ego_ptr, **kwargs):
        return self.model(x, adj_t, ego_ptr, **kwargs)

    def step(self, batch: Any):
        logits = self.forward(batch.x, batch.adj_t, batch.ego_ptr,
                              p=batch.p, batch=batch.batch, group_ptr=batch.group_ptr)
        loss = loss_fn(logits, batch.y)
        preds, y = pred_fn(logits, batch.y)
        return loss, preds, y

    def on_train_epoch_start(self) -> None:
        self.sampler.to(self.device)  # 采样阶段使用gpu加速
        # self.sampler.to('cpu')

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss/batch.batch_size, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets,
                'num_nodes': batch.num_nodes, 'hop': batch.hop.to(torch.float).mean().item()}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        hop = num_nodes = 0
        for output in outputs:
            hop += output['hop']
            num_nodes += output['num_nodes']
        hop /= len(outputs)
        num_nodes /= len(outputs)

        self.log('train/hop', round(hop, 2), prog_bar=True)
        self.log('train/nodes', round(num_nodes), prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.val_acc(preds, targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        self.log("val/acc", acc, prog_bar=True)

        self.val_acc_best.update(acc)
        best_acc = self.val_acc_best.compute()
        self.log("val/acc_best", best_acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets,
                'num_nodes': batch.num_nodes, 'num_edges': batch.num_edges,
                'hop': batch.hop.to(torch.float).mean().item()}

    def test_epoch_end(self, outputs: List[Any]):
        hop = num_nodes = num_edges = 0
        for output in outputs:
            hop += output['hop']
            num_nodes += output['num_nodes']
            num_edges += output['num_edges']
        hop /= len(outputs)
        num_nodes, num_edges = num_nodes / len(outputs),  num_edges / len(outputs)

        self.log("test/acc", self.test_acc.compute(), prog_bar=True)
        self.log('test/hop', round(hop, 2))
        self.log('test/nodes', round(num_nodes))
        self.log('test/edges', round(num_edges))

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        param_config = [{
            'params': self.model.parameters(),
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay
        }]

        if hasattr(self, 'sampler_conf'):
            param_config.append(self.sampler_conf)

        return torch.optim.Adam(param_config)
