import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet50(num_classes=num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()

        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
