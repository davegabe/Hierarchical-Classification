import torch.nn as nn
import torch.nn.functional as F
import torch
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


class Branch(nn.Module):
    def __init__(self):
        """
        Block 4 and block 5 of VGG16
        """
        super(Branch, self).__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_4(x)
        x = self.block_5(x)
        return x


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


class VGG16(nn.Module):
    def __init__(self, n_classes: list[int], n_branches: int = 1, device: str = "cpu"):
        """
        VGG16 model with n branches and coarse classifiers.

        Args:
            n_classes (list[int]): Number of classes for each coarse classifier.
            n_branches (int, optional): Number of branches. Defaults to 1.
        """
        super(VGG16, self).__init__()

        # 2 coarse classifiers + 1 fine classifier
        assert len(n_classes) == 3

        self.n_branches = n_branches

        # Block 1
        self.block_1 = nn.Sequential(
            Conv2dBlock(3, 64),
            Conv2dBlock(64, 64),
            MaxPool2dBlock()
        )
        self.block_1_size = 64 * 112 * 112
        self.coarse_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.block_1_size, n_classes[0])
        )

        # Block 2
        self.block_2 = nn.Sequential(
            Conv2dBlock(64, 128),
            Conv2dBlock(128, 128),
            MaxPool2dBlock()
        )
        self.block_2_size = 128 * 56 * 56
        self.coarse_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.block_2_size, n_classes[1])
        )

        # Block 3
        self.block_3 = nn.Sequential(
            Conv2dBlock(128, 256),
            Conv2dBlock(256, 256),
            Conv2dBlock(256, 256),
            MaxPool2dBlock()
        )

        # Branches
        self.branch_selector_size = sum(n_classes[:-1]) # Sum of coarse classifiers
        self.branch_selector = nn.Sequential(
            nn.Linear(self.branch_selector_size, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_branches),
            nn.Sigmoid()
        )
        self.branches = [Branch().to(device) for _ in range(n_branches)]

        # Fine classifier
        self.fine_size = 512 * 7 * 7
        self.fine = Classifier(self.fine_size, n_classes[-1])

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
        c1 = self.coarse_1(x) # Shape: (batch_size, n_classes[0])
        x = self.block_2(x)
        c2 = self.coarse_2(x) # Shape: (batch_size, n_classes[1])
        x = self.block_3(x)
         
        # Select best branch by coarse prediction
        coarses = torch.cat([c1, c2], dim=1) # Shape: (batch_size, n_classes[0] + n_classes[1])
        b_att: torch.Tensor = self.branch_selector(coarses) # Shape: (batch_size, n_branches)
        max_indices = torch.argmax(b_att, dim=1) # Shape: (batch_size,)
        
        # Apply branches
        batch_x = [self.branches[max_indices[i]](x[i].unsqueeze(0)) for i in range(x.shape[0])]
        x = torch.cat(batch_x, dim=0)

        # Concatenate branches
        x = self.avgpool(x)

        # Flatten and apply classifier
        fine = self.fine(x)

        # Concatenate coarse and fine predictions
        probas = torch.cat([c1, c2, fine], dim=1)
        return probas
