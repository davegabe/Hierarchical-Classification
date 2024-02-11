import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
import wandb
import numpy as np

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, dataloader: DataLoader, previous_size: list[int], device: torch.device):

    # WandB
    wandb.init(
        project="hierarchical-classification",
        config={
            "model": MODEL_NAME,
            "branch_selector": BRANCH_SELECTOR if MODEL_NAME == "branch_vgg16" else None,
            "n_branches": N_BRANCHES,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "limit_classes": LIMIT_CLASSES,
            "image_size": IMAGE_SIZE,
            "loss_weights": loss_fn.weights if MODEL_NAME=='branch_vgg16' else None 
        }
    )
    
    # Train the model
    for epoch in range(NUM_EPOCHS):
        # Loop over the dataset
        running_loss = 0.0
        epoch_accuracy = 0
        total_branch_choices = []

        for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move tensors to the configured device
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits  = model(images)
            loss = loss_fn(logits, labels)            

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            if MODEL_NAME == "branch_vgg16":
                accuracies = []
                for i in range(logits.shape[0]):
                    # Fine prediction accuracy
                    t = logits[i, previous_size[-2]:]  # Shape: (batch_size, size)
                    l = labels[i, previous_size[-2]:]  # Shape: (batch_size, size)
                    accuracies.append(torch.argmax(t) == torch.argmax(l))
                accuracies = torch.tensor(accuracies)
                epoch_accuracy += torch.sum(accuracies) / accuracies.shape[0]
                # If branch selector is learnable, add the l1 loss
                if BRANCH_SELECTOR == "learnable":
                    loss += model.l1_loss() * BS_L1_REGULARIZATION
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

        # Log to wandb the branch scores
        if MODEL_NAME == "branch_vgg16":
            branch_choices = model.branch_choices
            total_branch_choices += branch_choices
            model.reset_branch_choices()
            wandb.log({
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "histogram": branch_choices,
                "w_coarse_0": loss_fn.weights[0],
                "w_coarse_1": loss_fn.weights[1],
                "w_fine": loss_fn.weights[2]
            }, step=epoch)
            if BRANCH_SELECTOR == "learnable":
                # Log the branch selector weights for each output node
                weight_matrix = model.branch_selector.linear.weight.detach().cpu().numpy()
                for i, param in enumerate(weight_matrix):
                    wandb.log({f"bs_{i + 1}_weights": wandb.Histogram(param)}, step=epoch)
        else:
            wandb.log({"loss": epoch_loss, "accuracy": epoch_accuracy}, step=epoch)

    wandb.finish()


