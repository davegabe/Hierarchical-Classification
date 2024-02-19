import torch
import torch.optim as optim
import torch.nn as nn
import itertools
from modules.utils import accuracy_fn, loss_fn
from modules.vgg_light import Conv2dBlock, MaxPool2dBlock, Classifier
from config import BRANCH_SELECTOR, L1_REGULARIZATION, SIMILARITY_REGULARIZATION, LOG_STEP, PRIVILEGED
import wandb
import pytorch_lightning as L


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


class BranchVGG16(L.LightningModule):
    def __init__(self, n_classes: list[int], n_branches: int = 1, eps: float = 0, lr=1e-3):
        """
        VGG16 model with n branches and coarse classifiers.

        Args:
            n_classes (list[int]): Number of classes for each coarse classifier.
            n_branches (int, optional): Number of branches. Defaults to 1.
        """
        super(BranchVGG16, self).__init__()

        # 2 coarse classifiers + 1 fine classifier
        assert len(n_classes) == 3
        self.weights = [0.5, 0, 0.5]

        self.n_classes = n_classes
        self.n_branches = n_branches
        self.eps = eps
        self.lr = lr
        self.step = 0

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
        self.branches = [Branch() for _ in range(n_branches)]

        # Fine classifier
        self.fine_size = 512 * 7 * 7
        self.fine = Classifier(self.fine_size, n_classes[-1])

        # Branch selector
        self.coarse_classifier = nn.Sequential(
            Branch(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(self.fine_size, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, n_classes[0] + n_classes[1])
        )
        if BRANCH_SELECTOR == "learnable":
            self.branch_selector = BranchSelector(
                n_classes[0], n_branches
            )
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

    def forward(self, x: torch.Tensor, c1_true=None, training=False) -> tuple[torch.Tensor, torch.Tensor]:
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
        if PRIVILEGED and training:
            branch_scores = self.branch_selector(c1_true)
        else:
            branch_scores = self.branch_selector(c1)
        max_indices = torch.argmax(branch_scores, dim=1)

        # Apply selected branch
        results = []
        for i in range(x.shape[0]):
            max_index = max_indices[i]
            results.append(self.branches[max_index](x[i]))
            # Update branch choices
            self.branch_choices += [max_index]
        x = torch.stack(results)

        # Apply average pooling
        x = self.avgpool(x)

        # Flatten and apply classifiers
        fine = self.fine(x)

        # log steps
        if self.step % LOG_STEP == 0:
            if BRANCH_SELECTOR == "learnable":
                # Log the branch selector weights for each output node
                weight_matrix = self.branch_selector.linear.weight.detach().cpu().numpy()
                for i, param in enumerate(weight_matrix):
                    wandb.log({
                        f"bs_{i + 1}_weights": wandb.Histogram(param)
                    }, step=self.step)
        self.step += 1

        # Concatenate coarse and fine predictions
        return c1, c2, fine

    def regularize(self) -> torch.Tensor:
        """
        Compute regularization loss.

        Returns:
            torch.Tensor: Regularization loss.
        """
        return self.branch_selector.regularize()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        labels_arr = [
            labels[:, :self.n_classes[0]],
            labels[:, self.n_classes[0]:self.n_classes[0]+self.n_classes[1]],
            labels[:, self.n_classes[0]+self.n_classes[1]:]
        ]
        c1_true = labels_arr[-1]
        c1, c2, fine = self(images, c1_true, training=True)

        # Compute loss
        loss = loss_fn(self.weights, c1, c2, fine, labels_arr)

        # Compute accuracy
        accuracy = accuracy_fn(fine, labels_arr[-1])

        if BRANCH_SELECTOR == 'learnable':
            loss += self.regularize()

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
        accuracy = accuracy_fn(fine, labels_arr[-1])

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}
