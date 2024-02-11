import torch
import torch.nn as nn
import numpy as np
import wandb

class HierarchicalLoss(nn.Module):
    def __init__(self, hierarchy_size: list[int]):
        super(HierarchicalLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.hierarchy_size = hierarchy_size
        self.previous_size = [sum(hierarchy_size[:i]) for i in range(len(hierarchy_size))]
        self.weights = np.array([0.9, 0, 0.1])
        self.weights = self.weights / np.sum(self.weights)
        self.last_epoch = 0
        
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Compute the loss for each hierarchy level and return the weighted sum of losses.

        Args:
            logits (torch.Tensor): Logits.
            labels (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Loss.
        """
        # Update weights on epoch change
        if epoch != self.last_epoch:
            self.last_epoch = epoch
            self.weights[0] = max(self.weights[0] * 0.99, 0.1)
            self.weights[2] = 1 - self.weights[0] - self.weights[1]
            self.weights = self.weights / np.sum(self.weights)

        # Compute the loss for each level of the hierarchy
        losses = []
        for previous_size, size in zip(self.previous_size, self.hierarchy_size):
            t = logits[:, previous_size:previous_size+size] # Shape: (batch_size, size)
            l = labels[:, previous_size:previous_size+size] # Shape: (batch_size, size)
            losses.append(self.loss(t, l))

        # Weighted sum of losses
        total_loss = 0
        for i in range(len(losses)):
            total_loss += self.weights[i] * losses[i]

        return total_loss