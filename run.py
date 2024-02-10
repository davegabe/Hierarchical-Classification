import torch
from torch.utils.data import DataLoader
from modules.vgg import VGG16
from modules.branch_vgg import BranchVGG16
from modules.dataset import HierarchicalImageNet
from modules.loss import HierarchicalLoss
from train import train
from config import *


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = HierarchicalImageNet("train", only_leaves=MODEL_NAME == "vgg16")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Hierarchy classes size
    hierarchy_size = dataset.hierarchy_size
    previous_size = [sum(hierarchy_size[:i+1])
                     for i in range(len(hierarchy_size))]

    # Load the model and loss
    if MODEL_NAME == "vgg16":
        model = VGG16(n_classes=hierarchy_size[-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()
    elif MODEL_NAME == "branch_vgg16":
        model = BranchVGG16(n_classes=hierarchy_size, device=device,
                            n_branches=N_BRANCHES).to(device)
        loss = HierarchicalLoss(hierarchy_size)

    # Print the number of trainable parameters
    size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {size / 1e6:.3f} million (total: {size})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(model, optimizer, loss, dataloader, previous_size, device)


if __name__ == "__main__":
    main()
