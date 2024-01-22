import torch
from torch.utils.data import DataLoader
from config import *


def test(model: torch.nn.Module, dataloader: DataLoader, previous_size: list[int], device: torch.device):
    # Test the model
    for i, (images, labels) in enumerate(dataloader):
        # Move tensors to the configured device
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)

        # Forward pass
        logits = model(images)

        # TODO: Write test code here

    return
