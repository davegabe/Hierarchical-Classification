import torch.nn as nn
import torch.nn.functional as F
import torch
import itertools
from modules.vgg import Conv2dBlock, MaxPool2dBlock, Classifier
from config import BRANCH_SELECTOR, L1_REGULARIZATION, SIMILARITY_REGULARIZATION
import random


class BranchSelector(nn.Module):
    def __init__(self, n_classes: int, n_branches: int):
        """
        Learnable branch selector.

        Args:
            n_classes (int): Number of classes.
            n_branches (int): Number of branches.
        """
        super(BranchSelector, self).__init__()
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.linear = nn.Linear(n_classes, n_branches)
        self.activation = nn.Sigmoid()
        print(f"BranchSelector: {n_classes} -> {n_branches} branches")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.linear(x)
        x = self.activation(x)
        return x

    def regularize(self) -> torch.Tensor:
        """
        Compute L1 loss on the rows of the weight matrix and similiarity loss on the rows of the weight matrix.
        This encourages the model to use different branches for different classes.


        Returns:
            torch.Tensor: L1 loss.
        """
        device = self.linear.weight.device
        l1 = torch.tensor(0.).to(device)
        sim = torch.tensor(0.).to(device)
        weights = self.linear.weight

        # For each output neuron, compute the L1 norm of the weights and the similarity loss
        for i, param in enumerate(weights):
            l1 += torch.norm(param, 1)

        # Compute combination of weights to check similarity
        indexes = [i for i in range(self.n_branches)]
        combinations = list(itertools.combinations(indexes, 2))
        # # Compute similarity loss
        # for i, j in combinations:
        #     sim += F.cosine_similarity(weights[i], weights[j])
            
        return l1 * L1_REGULARIZATION + sim * SIMILARITY_REGULARIZATION


class NonLearnableBranchSelector(nn.Module):
    def __init__(self, n_branches: int):
        """
        Non-learnable branch selector.

        Args:
            n_branches (int): Number of branches.
        """
        super(NonLearnableBranchSelector, self).__init__()
        self.n_branches = n_branches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Initialize branch scores
        branch_scores = torch.zeros(x.shape[0], self.n_branches)
        # Compute the sum of the splits
        split = x.shape[1] // self.n_branches
        for i in range(self.n_branches):
            start = i * split
            end = max((i + 1) * split, x.shape[1])
            branch_scores[:, i] = torch.sum(x[:, start:end], dim=1)
        return branch_scores


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
    def __init__(self, n_classes: list[int], n_branches: int = 1, device: str = "cpu", eps: float = 0):
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
        self.eps = eps

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
        if BRANCH_SELECTOR == "learnable":
            self.branch_selector = BranchSelector(
                n_classes[0], n_branches
            ).to(device)
        else:
            self.branch_selector = NonLearnableBranchSelector(n_branches)

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def reset_branch_choices(self):
        self.branch_choices = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply blocks
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        # Select best branch by coarse prediction
        # Shape: (batch_size, n_classes[0] + n_classes[1])
        coarses = self.coarse_classifier(x)
        # Shape: (batch_size, n_classes[0])
        c1 = coarses[:, :self.n_classes[0]]
        # Shape: (batch_size, n_classes[1])
        c2 = coarses[:, self.n_classes[0]:]

        # Select branch
        branch_scores = self.branch_selector(c1)
        max_indices = torch.argmax(branch_scores, dim=1)

        # Apply selected branch
        results = []
        for i in range(x.shape[0]):
            if random.random() > self.eps:
                max_index = random.randint(0, self.n_branches - 1)
            else:
                max_index = max_indices[i]
            results.append(self.branches[max_index](x[i]))
            # Update branch choices
            self.branch_choices += [max_index]
        self.eps *= 0.99
        x = torch.stack(results)

        # Apply average pooling
        x = self.avgpool(x)

        # Flatten and apply classifier
        fine = self.fine(x)

        # Concatenate coarse and fine predictions
        probas = torch.cat([c1, c2, fine], dim=1)
        return probas

    def regularize(self) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Returns:
            torch.Tensor: Regularization loss.
        """
        return self.branch_selector.regularize()
