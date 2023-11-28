import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm = False):
        super(Conv2dBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
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


class VGG16(nn.Module):
    def __init__(self, n_classes: int):
        super(VGG16, self).__init__()

        self.block_1 = nn.Sequential(
            Conv2dBlock(3, 64),
            Conv2dBlock(64, 64),
            MaxPool2dBlock()
        )

        self.block_2 = nn.Sequential(
            Conv2dBlock(64, 128),
            Conv2dBlock(128, 128),
            MaxPool2dBlock()
        )

        self.block_3 = nn.Sequential(
            Conv2dBlock(128, 256),
            Conv2dBlock(256, 256),
            Conv2dBlock(256, 256),
            MaxPool2dBlock()
        )

        self.block_4 = nn.Sequential(
            Conv2dBlock(256, 512),
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            MaxPool2dBlock()
        )

        self.block_5 = nn.Sequential(
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            Conv2dBlock(512, 512),
            MaxPool2dBlock()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )

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
        x = self.avgpool(x)

        # Flatten and apply classifier
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)

        return logits, probas
