from modules.models import VGG16
from modules.dataset import HierarchicalImageNet
from modules.loss import HierarchicalLoss
import torch
from config import *

# Load the dataset
dataset = HierarchicalImageNet("train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Hierarchy classes size
hierarchy_size = dataset.hierarchy_size

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(n_classes=hierarchy_size).to(device)

# Define the loss and optimizer
hierarchial_loss = HierarchicalLoss(hierarchy_size)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Train the model
for epoch in range(NUM_EPOCHS):
    # Loop over the dataset
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        # Move tensors to the configured device
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)
        
        # Forward pass
        probas = model(images)
        loss = hierarchial_loss(probas, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        optimizer.step()
        
        # Sum of all losses 
        running_loss += loss.item() 
           
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss.item():.4f}')
    