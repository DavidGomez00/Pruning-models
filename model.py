import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision import models


class VGG16_for_catsndogs(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.lr = learning_rate
        # Modify VGG-16 model for our purpose
        self.model = models.vgg16(weights='DEFAULT')
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1)

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.train_outputs= []
        self.val_outputs = []

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({'train_loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        self.train_outputs.append({'loss': loss, 'scores': scores, 'y':y})
        return {'loss': loss, 'scores': scores, 'y':y}
    
    def on_train_epoch_end(self):
        scores = torch.cat([x["scores"] for x in self.train_outputs])
        y = torch.cat([x["y"] for x in self.train_outputs])
        self.train_outputs = []

        self.log_dict({
            "train_acc": self.accuracy(scores, y),
            "train_f1": self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True
        )

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict({'val_loss': loss},
                      on_step=False, on_epoch=True, prog_bar=True)
        self.val_outputs.append({'loss': loss, 'scores': scores, 'y':y})
        return {'loss': loss, 'scores': scores, 'y':y}
    
    def on_validation_epoch_end(self):
        scores = torch.cat([x["scores"] for x in self.val_outputs])
        y = torch.cat([x["y"] for x in self.val_outputs])
        self.val_outputs = []

        self.log_dict({
            "val_acc": self.accuracy(scores, y),
            "val_f1": self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True
        )
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = torch.squeeze(self.model.forward(x), dim=1)
        loss = self.loss_fn(scores, y.float())
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = torch.squeeze(self.model.forward(x), dim=1)
        preds = torch.argmax(scores, dim=1)
        return preds
        
    def configure_optimizers(self):
        # Filter the parameters based on `requires_grad`
        return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)