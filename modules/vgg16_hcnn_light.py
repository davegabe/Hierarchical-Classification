import pytorch_lightning as L
import torch
import torch.nn as nn
from modules.utils import accuracy_fn, loss_fn
from modules.vgg_light import *
from config import *


class CoarseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super(CoarseBlock, self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


class VGG16_HCNN(L.LightningModule):
    def __init__(self, n_classes, lr=1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weights = [0.8, 0.1, 0.1]

        self.block1 = nn.Sequential(
            Conv2dBlock(3, 64, batch_norm=True),
            Conv2dBlock(64, 64, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block2 = nn.Sequential(
            Conv2dBlock(64, 128, batch_norm=True),
            Conv2dBlock(128, 128, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block3 = nn.Sequential(
            Conv2dBlock(128, 256, batch_norm=True),
            Conv2dBlock(256, 256, batch_norm=True),
            Conv2dBlock(256, 256, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block4 = nn.Sequential(
            Conv2dBlock(256, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block5 = nn.Sequential(
            Conv2dBlock(512, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            MaxPool2dBlock()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.coarse1_block = CoarseBlock(128 * 32 * 32, n_classes[0])
        self.coarse2_block = CoarseBlock(256 * 16 * 16, n_classes[1])
        self.classifier = Classifier(512 * 7 * 7, n_classes[2])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        c1 = self.coarse1_block(x)
        x = self.block3(x)
        c2 = self.coarse2_block(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        fine = self.classifier(x)
        return c1, c2, fine

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        c1, c2, fine = self(images)
        labels_arr = [
            labels[:, 0:c1.shape[1]],
            labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]],
            labels[:, c1.shape[1]+c2.shape[1]:]
        ]

        # Compute loss
        loss = loss_fn(self.weights, c1, c2, fine, labels_arr)

        # Compute accuracy
        accuracy = accuracy_fn(fine, labels_arr)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        c1, c2, fine = self(images)
        labels_arr = [
            labels[:, 0:c1.shape[1]],
            labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]],
            labels[:, c1.shape[1]+c2.shape[1]:]
        ]

        # Compute loss
        loss = loss_fn(self.weights, c1, c2, fine, labels_arr)

        # Compute accuracy
        accuracy = accuracy_fn(fine, labels_arr)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 5:
            self.weights = [0.1, 0.8, 0.1]
        elif self.current_epoch == 10:
            self.weights = [0.1, 0.1, 0.8]
