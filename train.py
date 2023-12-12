from modules.models import VGG16
from modules.dataset import HierarchicalImageNet
import torch
from config import *

# Load the dataset
dataset = HierarchicalImageNet("train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the model
model = VGG16(n_classes=[5,3,3])

# Define the loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Train the model
for epoch in range(NUM_EPOCHS):
    # Loop over the dataset
    for i, (images, labels) in enumerate(dataloader):
        # Forward pass
        logits, probas = model(images)
        # print(logits)
        # print(probas)

        # Backward and optimize
        optimizer.zero_grad()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}]')