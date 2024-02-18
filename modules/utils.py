import torch
from torch.nn import functional as F


def loss_fn(weights: list[float], c1: torch.Tensor, c2: torch.Tensor, fine: torch.Tensor, labels_arr: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute the loss function for the hierarchical classification model.

    Args:
        weights (list[float]): List of weights for the three classifiers.
        c1 (Tensor): Output of the first classifier.
        c2 (Tensor): Output of the second classifier.
        fine (Tensor): Output of the fine classifier.
        labels_arr (list[Tensor]): List of labels for the three classifiers.

    Returns:
        Tensor: Loss value.
    """
    loss_c1 = F.cross_entropy(c1, labels_arr[0]) * weights[0]
    loss_c2 = F.cross_entropy(c2, labels_arr[1]) * weights[1]
    loss_fine = F.cross_entropy(fine, labels_arr[2]) * weights[2]
    return loss_c1 + loss_c2 + loss_fine


def accuracy_fn(fine: torch.Tensor, labels_arr: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute the accuracy of the fine classifier.

    Args:
        fine (Tensor): Output of the fine classifier.
        labels_arr (list[Tensor]): List of labels for the three classifiers.

    Returns:
        Tensor: Accuracy value.
    """
    fine_preds = torch.argmax(fine, dim=1)
    fine_labels = torch.argmax(labels_arr[2], dim=1)
    return torch.sum(fine_preds == fine_labels).float() / fine_preds.shape[0]
