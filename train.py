import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, dataloader: DataLoader, previous_size: list[int], device: torch.device):
    # Train the model
    for epoch in range(NUM_EPOCHS):
        # Loop over the dataset
        running_loss = 0.0
        epoch_accuracy = 0
        for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move tensors to the configured device
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(images)
            loss = loss_fn(logits, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            if MODEL_NAME == "branch_vgg16":
                accuracies = []
                for i in range(logits.shape[0]):
                    # Fine prediction accuracy
                    t = logits[i, previous_size[-1]:]  # Shape: (batch_size, size)
                    l = labels[i, previous_size[-1]:]  # Shape: (batch_size, size)
                    # print(max(t), min(t))
                    accuracies.append(torch.argmax(t) == torch.argmax(l))
                accuracies = torch.tensor(accuracies)
                epoch_accuracy += torch.sum(accuracies) / accuracies.shape[0]
            elif MODEL_NAME == "vgg16":
                correct = torch.sum(torch.argmax(logits, dim=1) == labels)
                epoch_accuracy += correct / logits.shape[0]

            # Sum of all losses
            running_loss += loss.item()

        # print accuracy per epoch
        epoch_accuracy /= len(dataloader)
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')
        print(f'Accuracy_epoch: {epoch_accuracy:.4f}')
