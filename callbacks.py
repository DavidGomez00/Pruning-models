import lightning as L
from lightning.pytorch.callbacks import Callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print('Start training!')

    def on_train_end(self, trainer, pl_module):
        print('Training stopped!')
    

class FeatureExtractorFreezeUnfreeze(L.pytorch.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self.__unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module
        self.freeze(pl_module.model.features)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When 'current_epoch' is 10, feature extractor will start training
        if current_epoch == self.__unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.features,
                optimizer = optimizer,
                train_bn = True,
            )

