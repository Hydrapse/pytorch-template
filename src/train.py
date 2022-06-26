from typing import List, Optional

import hydra
import torch.backends.cudnn
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.datamodules.adapt_datamodule import AdaptDataModule

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    if isinstance(datamodule, AdaptDataModule):
        config.model.num_groups = config.datamodule.num_groups
        config.model.as_single_layer = config.datamodule.to_single_layer
        model: LightningModule = hydra.utils.instantiate(config.model,
                                                         sampler=datamodule.sampler,
                                                         in_channels=datamodule.num_features,
                                                         out_channels=datamodule.num_classes)
        model.sampler_conf = datamodule.optim_conf
    else:
        model: LightningModule = hydra.utils.instantiate(config.model,
                                                         in_channels=datamodule.num_features,
                                                         out_channels=datamodule.num_classes)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Log best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    [logg.log_text('model/best_model_path', best_model_path) for logg in logger]

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        # retrieve best model
        best_model = model.load_from_checkpoint(best_model_path)
        datamodule.sampler = best_model.sampler
        trainer.test(model=best_model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
