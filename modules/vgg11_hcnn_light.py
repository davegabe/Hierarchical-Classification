import pytorch_lightning as L
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modules.vgg import *
from modules.dataset import HierarchicalImageNet
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

class VGG11_HCNN(L.LightningModule):
    def __init__(self, n_classes, lr=1e-3) -> None:
        super().__init__()

        self.lr = lr
        self.weights = [0.8,0.1,0.1]

        self.block1 = nn.Sequential(
            Conv2dBlock(3, 64, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block2 = nn.Sequential(
            Conv2dBlock(64, 128, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block3 = nn.Sequential(
            Conv2dBlock(128, 256, batch_norm=True),
            Conv2dBlock(256, 256, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block4 = nn.Sequential(
            Conv2dBlock(256, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            MaxPool2dBlock()
        )

        self.block5 = nn.Sequential(
            Conv2dBlock(512, 512, batch_norm=True),
            Conv2dBlock(512, 512, batch_norm=True),
            MaxPool2dBlock()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.coarse1_block = CoarseBlock(128 * 56 * 56, n_classes[0])
        self.coarse2_block = CoarseBlock(256 * 28 * 28, n_classes[1])
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
        labels_arr = [labels[:, 0:c1.shape[1]], labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]], labels[:, c1.shape[1]+c2.shape[1]:]]
        loss = self.weights[0]*F.cross_entropy(c1, labels_arr[0]) + self.weights[1]*F.cross_entropy(c2, labels_arr[1]) + self.weights[2]*F.cross_entropy(fine, labels_arr[2])
        accuracies = []
        for i in range(fine.shape[0]):
            # Fine prediction accuracy
            t = fine[i]  # Shape: (batch_size, size)
            l = labels_arr[2][i]  # Shape: (batch_size, size)
            accuracies.append(torch.argmax(t) == torch.argmax(l))
        accuracies = torch.tensor(accuracies)
        accuracy = torch.sum(accuracies) / accuracies.shape[0]
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'loss':loss, 'accuracy':accuracy}
    
    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        c1, c2, fine = self(images)
        labels_arr = [labels[:, 0:c1.shape[1]], labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]], labels[:, c1.shape[1]+c2.shape[1]:]]
        accuracies = []
        for i in range(fine.shape[0]):
            # Fine prediction accuracy
            t = fine[i]
            l = labels_arr[2][i]
            accuracies.append(torch.argmax(t) == torch.argmax(l))
        accuracies = torch.tensor(accuracies)
        accuracy = torch.sum(accuracies) / accuracies.shape[0]
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'accuracy':accuracy}
    
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 5:
            self.weights = [0.1,0.8,0.1]
        elif self.current_epoch == 10:
            self.weights = [0.1,0.1,0.8]