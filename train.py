import torch
from torch.utils.data import DataLoader
from modules.vgg import VGG16
from modules.branch_vgg import BranchVGG16
from modules.loss import HierarchicalLoss
from modules.hcnn_loss import HCNNLoss
from modules.vgg11_hcnn import VGG11_HCNN
from tqdm import tqdm
from config import NUM_EPOCHS, BRANCH_SELECTOR, VAL_EPOCHS, BATCH_SIZE
import wandb

def train_vgg(model: VGG16, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.CrossEntropyLoss, dataloader: DataLoader, val_loader: DataLoader, device: torch.device):
    # Train the model
    for epoch in range(NUM_EPOCHS):
        # Loop over the dataset
        running_loss = 0.0
        epoch_accuracy = 0

        with tqdm(total=len(dataloader)*BATCH_SIZE) as pbar:
            for i, (images, labels) in enumerate(dataloader):
                # Move tensors to the configured device
                images: torch.Tensor = images.to(device)
                labels: torch.Tensor = labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                logits  = model(images)
                loss = loss_fn(logits, labels)

                # Compute accuracy
                correct = torch.sum(torch.argmax(logits, dim=1) == labels)
                epoch_accuracy += correct / logits.shape[0] 

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Sum of all losses
                running_loss += loss.item()
                pbar.update(BATCH_SIZE)

        # Print accuracy per epoch
        epoch_accuracy /= len(dataloader)
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')
        print(f'Accuracy_epoch: {epoch_accuracy:.4f}')

        # Validate the model
        if epoch % VAL_EPOCHS == 0:
            val_accuracy = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    images: torch.Tensor = images.to(device)
                    labels: torch.Tensor = labels.to(device)
                    logits  = model(images)
                    correct = torch.sum(torch.argmax(logits, dim=1) == labels)
                    val_accuracy += correct / logits.shape[0]
            val_accuracy /= len(val_loader)
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            wandb.log({"val_accuracy": val_accuracy}, step=epoch)

            # save model
            torch.save(model.state_dict(), f"models/vgg16_{epoch}.pt")

        # Log to wandb the branch scores
        wandb.log({"loss": epoch_loss, "accuracy": epoch_accuracy}, step=epoch)


def train_branch_vgg(model: BranchVGG16, optimizer: torch.optim.Optimizer, loss_fn: HierarchicalLoss, dataloader: DataLoader, val_loader: DataLoader, device: torch.device, previous_size: list):
    # Train the model
    for epoch in range(NUM_EPOCHS):
        # Loop over the dataset
        running_loss = 0.0
        epoch_accuracy = 0
        total_branch_choices = []

        with tqdm(total=len(dataloader)*BATCH_SIZE) as pbar:

            for i, (images, labels) in enumerate(dataloader):
                # Move tensors to the configured device
                images: torch.Tensor = images.to(device)
                labels: torch.Tensor = labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                logits  = model(images)
                loss = loss_fn(logits, labels, epoch)

                # Compute accuracy
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
                    loss += model.regularize()

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Sum of all losses
                running_loss += loss.item()
                pbar.update(BATCH_SIZE)

        # print accuracy per epoch
        epoch_accuracy /= len(dataloader)
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')
        print(f'Accuracy_epoch: {epoch_accuracy:.4f}')

        # Log to wandb the branch scores
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

        # Validate the model
        if epoch % VAL_EPOCHS:
            # TODO: Validate the model
            pass

def train_hcnn(model: VGG11_HCNN, optimizer: torch.optim.Optimizer, loss_fn: HierarchicalLoss, dataloader: DataLoader, val_loader: DataLoader, device: torch.device):
    # Train the model
    for epoch in range(NUM_EPOCHS):
        # Loop over the dataset
        running_loss = 0.0
        epoch_accuracy = 0
        total_branch_choices = []
        val_accuracy = 0


        with tqdm(total=len(dataloader)*BATCH_SIZE) as pbar:

            for i, (images, labels) in enumerate(dataloader):
                # Move tensors to the configured device
                images: torch.Tensor = images.to(device)
                labels: torch.Tensor = labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                c1, c2, fine = model(images)
                labels_arr = [labels[:, 0:c1.shape[1]], labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]], labels[:, c1.shape[1]+c2.shape[1]:]]
                loss = loss_fn([c1, c2, fine], labels_arr, epoch)

                # Compute accuracy
                accuracies = []
                for i in range(fine.shape[0]):
                    # Fine prediction accuracy
                    t = fine[i]  # Shape: (batch_size, size)
                    l = labels_arr[2][i]  # Shape: (batch_size, size)
                    accuracies.append(torch.argmax(t) == torch.argmax(l))
                accuracies = torch.tensor(accuracies)
                epoch_accuracy += torch.sum(accuracies) / accuracies.shape[0]
              
                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Sum of all losses
                running_loss += loss.item()
                pbar.update(BATCH_SIZE)

        # print accuracy per epoch
        epoch_accuracy /= len(dataloader)
        epoch_loss = running_loss / len(dataloader)
        print(f'Train Accuracy_epoch: {epoch_accuracy:.4f}')

        if epoch % VAL_EPOCHS == 0:
            with torch.no_grad():
                with tqdm(total=len(val_loader)*BATCH_SIZE) as pbar:
                    for i, (images, labels) in enumerate(val_loader):
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        c1, c2, fine = model(images)
                        labels_arr = [labels[:, 0:c1.shape[1]], labels[:, c1.shape[1]:c1.shape[1]+c2.shape[1]], labels[:, c1.shape[1]+c2.shape[1]:]]
                        accuracies = []
                        for i in range(fine.shape[0]):
                            # Fine prediction accuracy
                            t = fine[i]
                            l = labels_arr[2][i]
                            accuracies.append(torch.argmax(t) == torch.argmax(l))
                        accuracies = torch.tensor(accuracies)
                        val_accuracy += torch.sum(accuracies) / accuracies.shape[0]
                        pbar.update(BATCH_SIZE)
            val_accuracy /= len(val_loader)
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            wandb.log({"val_accuracy": val_accuracy}, step=epoch)
       
        wandb.log({
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "w_coarse_0": loss_fn.weights[0],
            "w_coarse_1": loss_fn.weights[1],
            "w_fine": loss_fn.weights[2]
        }, step=epoch)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

        