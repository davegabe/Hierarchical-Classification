from modules.models import VGG16
from modules.dataset import HierarchicalImageNet
import torch
from config import *

# Load the dataset
dataset = HierarchicalImageNet("train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the model
model = VGG16(n_classes=dataset.n_classes)

# Define the loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
loss = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(NUM_EPOCHS):
    pass