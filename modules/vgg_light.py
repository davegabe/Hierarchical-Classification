import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.vgg_light import *
from config import *


class Classifier(nn.Module):
    def __init__(self, input: int,  n_classes: int):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm=False):
        super(Conv2dBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.relu(x)


class MaxPool2dBlock(nn.Module):
    def __init__(self):
        super(MaxPool2dBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.maxpool(x)


class VGG16(L.LightningModule):
    def __init__(self, n_classes: int, lr=1e-3):
        """
        VGG16 model architecture.

        Args:
            n_classes (int): Number of classes.
        """
        super().__init__()

        self.lr = lr

        # Block 1
        self.block_1 = nn.Sequential(
            Conv2dBlock(3, 64),
            Conv2dBlock(64, 64),
            MaxPool2dBlock()
        )

        # Block 2
        self.block_2 = nn.Sequential(
            Conv2dBlock(64, 128),
            Conv2dBlock(128, 128),
            MaxPool2dBlock()
        )

        # Block 3
        self.block_3 = nn.Sequential(
            Conv2dBlock(128, 256),
            Conv2dBlock(256, 256),
            Conv2dBlock(256, 256),
            MaxPool2dBlock()
        )

        # Block 4
        self.block_4 = nn.Sequential(
            Conv2dBlock(256, 512),
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            MaxPool2dBlock()
        )

        # Block 5
        self.block_5 = nn.Sequential(
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            MaxPool2dBlock()
        )

        # Fine classifier
        self.fine = Classifier(512 * 7 * 7, n_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        # Apply average pooling
        x = self.avgpool(x)

        # Flatten and apply classifier
        fine = self.fine(x)

        return fine

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        correct = torch.sum(torch.argmax(logits, dim=1) == labels)
        accuracy = correct / logits.shape[0]

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        correct = torch.sum(torch.argmax(logits, dim=1) == labels)
        accuracy = correct / logits.shape[0]

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}
