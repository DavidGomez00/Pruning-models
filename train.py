import random

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from torch import cuda

import config
from callbacks import FeatureExtractorFreezeUnfreeze, MyPrintingCallback
from dataset import DogsAndCatsDataModule
from model import VGG16_for_catsndogs

'''
VGG-16: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html  
Dataset: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data
'''

if __name__ == "__main__":

    device = 'gpu' if cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('medium') # To make lightning happy

    ############################# Reproducibility ############################
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    ##########################################################################



    ###################################### Experiment################################################################
    # Deine Logger and Profiler objects
    logger = TensorBoardLogger("tb_logs", "VGG16_catsndogs_ft")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/VGG16_catsndogs_ft"),
        trace_memory = True,
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )

    for k in range(config.NUM_FOLDS):
        # Model
        model = VGG16_for_catsndogs(learning_rate=config.LEARNING_RATE)
        # Datamodule
        datamodule = DogsAndCatsDataModule(data_dir=config.DATA_DIR,
                                            batch_size=config.BATCH_SIZE,
                                            num_workers=config.NUM_WORKERS,
                                            folds=config.NUM_FOLDS,
                                            k=k,
                                            split_seed=config.SPLIT_SEED)
        
        # Define Trainer object
        trainer = L.Trainer(accelerator=config.ACCELERATOR,
                            devices=config.DEVICES,
                            min_epochs=config.MIN_EPOCHS,
                            max_epochs=config.NUM_EPOCHS,
                            precision=config.PRECISION,
                            logger=logger,
                            callbacks=[FeatureExtractorFreezeUnfreeze(),
                                    EarlyStopping(monitor="train_loss", min_delta=0.001, patience=4),
                                    MyPrintingCallback(),
                                    ModelCheckpoint(dirpath="model_ckpt/",
                                                    filename='VGG16_ctsndgs_'+str(k)+'_{epoch}-{val_loss:.2f}',
                                                    monitor="val_loss",
                                                    save_weights_only=True)],
                            log_every_n_steps=16)

        trainer.fit(model, datamodule)
    #############################################################################################################


