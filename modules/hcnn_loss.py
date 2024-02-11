import torch
import torch.nn as nn
import numpy as np
import wandb

class HCNNLoss(nn.Module):
    def __init__(self, hierarchy_size: list[int]):
        super(HCNNLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.weights = np.array([0.98, 0.01, 0.01])
        self.weights = self.weights / np.sum(self.weights)
        self.last_epoch = 0
        
        
    def forward(self, preds: list[torch.Tensor, torch.Tensor, torch.Tensor], labels: list[torch.Tensor, torch.Tensor, torch.Tensor], epoch: int) -> torch.Tensor:
        """
        Compute the loss for each hierarchy level and return the weighted sum of losses.

        Args:
            logits (torch.Tensor): Logits.
            labels (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Loss.
        """

        if epoch > 25:
            self.weights = np.array([0.1, 0.8, 0.1])
            self.weights = self.weights / np.sum(self.weights)
        
        elif epoch > 50:
            self.weights = np.array([0.1, 0.1, 0.8])
            self.weights = self.weights / np.sum(self.weights)

        # Compute the loss for each level of the hierarchy
        losses = []
        c1 = preds[0]
        c2 = preds[1]
        fine = preds[2]

        # Coarse 1
        loss_c1 = self.loss(c1, labels[0])
        losses.append(loss_c1)
        # Coarse 2
        loss_c2 = self.loss(c2, labels[1])
        losses.append(loss_c2)
        # Fine
        loss_fine = self.loss(fine, labels[2])
        losses.append(loss_fine)

        # Weighted sum of losses
        total_loss = 0
        for i in range(len(losses)):
            total_loss += self.weights[i] * losses[i]

        return total_loss