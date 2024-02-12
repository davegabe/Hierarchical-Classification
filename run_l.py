import torch
import pytorch_lightning as L
from modules.vgg11_hcnn_light import VGG11_HCNN
from modules.vgg_light import VGG16
from modules.branch_vgg_light import BranchVGG16
from modules.dataset import HierarchicalImageNet
from torch.utils.data import DataLoader
from config import *

    

if __name__ == "__main__":
    
    train_dataset = HierarchicalImageNet(split='/home/riccardo/Documents/train', only_leaves= MODEL_NAME=='vgg16')
    val_dataset = HierarchicalImageNet(split='./dataset/validation', only_leaves= MODEL_NAME=='vgg16')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=11)


   

    trainer = L.Trainer(max_epochs=NUM_EPOCHS, benchmark=True, default_root_dir='./models', enable_checkpointing=True)

    with trainer.init_module():
        if MODEL_NAME == "vgg11_hcnn":
            model = VGG11_HCNN(n_classes=train_dataset.hierarchy_size, lr=LEARNING_RATE)
        elif MODEL_NAME == "vgg16":
            model = VGG16(n_classes=train_dataset.hierarchy_size[-1], lr=LEARNING_RATE)
        elif MODEL_NAME == "branch_vgg16":
            model = BranchVGG16(n_classes=train_dataset.hierarchy_size, n_branches=N_BRANCHES, eps=0, lr=LEARNING_RATE)


    trainer.fit(model, train_loader, val_loader)