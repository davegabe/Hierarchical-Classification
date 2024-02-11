import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
from modules.vgg import Conv2dBlock, MaxPool2dBlock, Classifier

class CoarseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels):
        super(CoarseBlock, self).__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


class VGG11_HCNN(nn.Module):
    def __init__(self, n_classes) -> None:
        super(VGG11_HCNN, self).__init__()

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
        # return x