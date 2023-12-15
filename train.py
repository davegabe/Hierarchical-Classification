from modules.models import VGG16
from modules.dataset import HierarchicalImageNet
from modules.loss import HierarchicalLoss
import torch
from tqdm import tqdm
from config import *

# Load the dataset
dataset = HierarchicalImageNet("train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hierarchy classes size
hierarchy_size = dataset.hierarchy_size
previous_size = [sum(hierarchy_size[:i]) for i in range(len(hierarchy_size))]

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(n_classes=hierarchy_size, device=device, n_branches=1).to(device)

# Define the loss and optimizer
hierarchial_loss = HierarchicalLoss(hierarchy_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(len(dataloader))
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
        loss = hierarchial_loss(logits, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        accuracies = []
        for i in range(logits.shape[0]):
            # Fine prediction accuracy
            t = logits[i, previous_size[-1]:] # Shape: (batch_size, size)
            l = labels[i, previous_size[-1]:] # Shape: (batch_size, size)
            # print(max(t), min(t))
            accuracies.append(torch.argmax(t) == torch.argmax(l))
        accuracies = torch.tensor(accuracies)
        epoch_accuracy += torch.sum(accuracies) / accuracies.shape[0]
        
        # Sum of all losses 
        running_loss += loss.item() 

    # print accuracy per epoch
    epoch_accuracy /= len(dataloader)       
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}', f'Accuracy_epoch: {epoch_accuracy:.4f}')
    