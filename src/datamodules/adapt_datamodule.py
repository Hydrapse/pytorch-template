from typing import Optional

from pytorch_lightning import LightningDataModule

from src.datamodules.components.data import get_data
from src.datamodules.components.loader import EgoGraphLoader
from src.datamodules.components.sampler import AdaptiveSampler
from src.utils.index import Dict


class AdaptDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            dataset: str,
            data_dir: str = "data/",
            split: str = 'full',
            batch_size: int = 64,
            undirected: bool = False,
            pin_memory: bool = False,
            num_workers: int = 0,
            persistent_workers: bool = False,
            # sampler
            lr: float = 0.01,
            weight_decay: float = 1e-4,
            **sampler_kwargs
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        data, num_features, num_classes, processed_dir = get_data(dataset, data_dir, split=split)

        self.data = data
        self.num_classes = num_classes
        self.processed_dir = processed_dir

        self.sampler = AdaptiveSampler(data, **sampler_kwargs)
        self.num_features = self.sampler.feature_size

        self.optim_conf = Dict({
            'params': self.sampler.parameters(),
            'lr': lr,
            'weight_decay': weight_decay
        })

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load components only if they're not loaded already
        pass

    def train_dataloader(self):
        return EgoGraphLoader(
            self.data.train_mask,
            self.sampler,
            self.hparams.undirected,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=0,
        )

    def val_dataloader(self):
        return EgoGraphLoader(
            self.data.val_mask,
            self.sampler,
            self.hparams.undirected,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers
        )

    def test_dataloader(self):
        return EgoGraphLoader(
            self.data.val_mask,
            self.sampler,
            self.hparams.undirected,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers
        )
