import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.vgg import Conv2dBlock, MaxPool2dBlock, Classifier
from config import *

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



class BranchVGG16(nn.Module):
    def __init__(self, n_classes: list[int], n_branches: int = 1, device: str = "cpu"):
        """
        VGG16 model with n branches and coarse classifiers.

        Args:
            n_classes (list[int]): Number of classes for each coarse classifier.
            n_branches (int, optional): Number of branches. Defaults to 1.
        """
        super(BranchVGG16, self).__init__()

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
