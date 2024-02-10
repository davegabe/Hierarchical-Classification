import torch.nn as nn
import torch.nn.functional as F
import torch
from modules.vgg import Conv2dBlock, MaxPool2dBlock, Classifier
from config import *
import wandb

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

        self.n_classes = n_classes
        self.n_branches = n_branches

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

        # Branches
        self.branches = [Branch().to(device) for _ in range(n_branches)]

        # Fine classifier
        self.fine_size = 512 * 7 * 7
        self.fine = Classifier(self.fine_size, n_classes[-1])

        # Branch selector
        self.coarse_classifier = nn.Sequential(
            Branch().to(device),
            nn.Flatten(),
            nn.Linear(self.fine_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, n_classes[0] + n_classes[1])
        )
        self.branch_selector = nn.Sequential(
            nn.Linear(n_classes[1], n_branches),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.branch_choices = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight)
                # nn.init.constant_(m.bias, 0)

    def reset_branch_choices(self):
        self.branch_choices = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # Select best branch by coarse prediction
        coarses = self.coarse_classifier(x) # Shape: (batch_size, n_classes[0] + n_classes[1])
        c1 = coarses[:, :self.n_classes[0]] # Shape: (batch_size, n_classes[0])
        c2 = coarses[:, self.n_classes[0]:] # Shape: (batch_size, n_classes[1])
        branch_scores = self.branch_selector(c2)
        max_indices = torch.argmax(branch_scores, dim=1)

        # Update branch choices
        self.branch_choices += max_indices.detach().cpu().tolist()

        # Apply selected branch
        results = []
        for i in range(x.shape[0]):
            max_index = max_indices[i]
            results.append(self.branches[max_index](x[i]))
        x = torch.stack(results)

        # Flatten and apply classifier
        fine = self.fine(x)

        # Concatenate coarse and fine predictions
        probas = torch.cat([c1, c2, fine], dim=1)
        return probas, branch_scores
