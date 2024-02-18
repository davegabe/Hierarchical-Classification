import torch
from torch.utils.data import DataLoader
from modules.vgg11_hcnn import VGG11_HCNN
from modules.hcnn_loss import HCNNLoss
from modules.vgg import VGG16
from modules.branch_vgg import BranchVGG16
from modules.dataset import HierarchicalImageNet
from modules.loss import HierarchicalLoss
from train import train_vgg, train_branch_vgg, train_hcnn
from config import MODEL_NAME, BRANCH_SELECTOR, N_BRANCHES, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, LIMIT_CLASSES, IMAGE_SIZE
from torchinfo import summary
import wandb 
import nltk
nltk.download('wordnet')

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path= "ILSVRC2012_img_train" # Change this to the path of the training set
    val_path = "ILSVRC2012_img_val" # Change this to the path of the validation set

    # in_hier = ImageNetHierarchy(in_path, in_info_path)

    # Load the dataset
    dataset = HierarchicalImageNet(split=train_path, only_leaves=MODEL_NAME == "vgg16")
    val_dataset = HierarchicalImageNet(split=val_path, only_leaves=MODEL_NAME == "vgg16")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    # Hierarchy classes size
    hierarchy_size = [x for x in dataset.hierarchy_size]
    previous_size = [sum(hierarchy_size[:i+1])
                     for i in range(len(hierarchy_size))]
    

    # Load the model and loss
    if MODEL_NAME == "vgg16":
        model = VGG16(n_classes=hierarchy_size[-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()
        # model = vgg16_bn().to(device)
        # loss = torch.nn.CrossEntropyLoss()
    elif MODEL_NAME == "branch_vgg16":
        model = BranchVGG16(n_classes=hierarchy_size, device=device,
                            n_branches=N_BRANCHES).to(device)
        loss = HierarchicalLoss(hierarchy_size)
    elif MODEL_NAME == "vgg11_hcnn":
        model = VGG11_HCNN(n_classes=hierarchy_size).to(device)
        loss = HCNNLoss(hierarchy_size)


    # Print the number of trainable parameters
    size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {size / 1e6:.3f} million (total: {size})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # summary(model, input_size=(BATCH_SIZE, 3, *IMAGE_SIZE))

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
            "loss_weights": loss.weights if MODEL_NAME=='branch_vgg16' else None 
        }
    )


    # Train the model
    if MODEL_NAME == "vgg16":
        train_vgg(model, optimizer, loss, dataloader, val_loader, device)
    elif MODEL_NAME == "branch_vgg16":
        train_branch_vgg(model, optimizer, loss, dataloader, device, previous_size)
    elif MODEL_NAME == "vgg11_hcnn":
        train_hcnn(model, optimizer, loss, dataloader, val_loader, device)

    # WandB
    wandb.finish()


if __name__ == "__main__":
    main()
