import torch
import pytorch_lightning as L
from modules.condhresnet_light import CondHResNet
from modules.vgg11_hcnn_light import VGG11_HCNN
from modules.vgg_light import VGG16
from modules.vgg16_hcnn_light import VGG16_HCNN
from modules.branch_vgg_light import BranchVGG16
from modules.dataset import HierarchicalImageNet
from modules.resnet_light import ResNetClassifier
from modules.hresnet_light import HResNet
from torch.utils.data import DataLoader
from config import *
import random
import wandb
import numpy as np
from pytorch_lightning.callbacks import LearningRateFinder, EarlyStopping, RichProgressBar
from torchinfo import summary
torch.random.manual_seed(42)
np.random.seed(42)
random.seed(42)
    
torch.set_float32_matmul_precision('medium')
if __name__ == "__main__":
    
    random_state = 42

    train_dataset = HierarchicalImageNet(split=TRAIN_DATASET_PATH, only_leaves= (MODEL_NAME=='vgg16' or MODEL_NAME=='resnet'),random_state=random_state)
    val_dataset = HierarchicalImageNet(split=VAL_DATASET_PATH, only_leaves= (MODEL_NAME=='vgg16' or MODEL_NAME=='resnet'),random_state=random_state)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=11)


    if USE_WANDB:
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
                # "loss_weights": loss.weights if MODEL_NAME=='branch_vgg16' else None 
            }
        )


    loggers = [L.loggers.WandbLogger(project="hierarchical-classification"), L.loggers.TensorBoardLogger("models/lightning_logs", name=MODEL_NAME)]



    trainer = L.Trainer(
        logger=loggers,
        max_epochs=NUM_EPOCHS, 
        benchmark=True, 
        default_root_dir='./models', 
        enable_checkpointing=True, 
        # precision="bf16-mixed", 
        accumulate_grad_batches=4,
        callbacks=[
            LearningRateFinder(min_lr=1e-5, max_lr=1e-1),
            # EarlyStopping('val_loss', mode='min', min_delta='0.005')
            RichProgressBar()
            ]
        )


    with trainer.init_module():
        if MODEL_NAME == "vgg11_hcnn":
            model = VGG11_HCNN(n_classes=train_dataset.hierarchy_size, lr=LEARNING_RATE)
        elif MODEL_NAME == "vgg16":
            model = VGG16(n_classes=train_dataset.hierarchy_size[-1], lr=LEARNING_RATE)
        elif MODEL_NAME == "vgg16_hcnn":
            model = VGG16_HCNN(n_classes=train_dataset.hierarchy_size, lr=LEARNING_RATE)
        elif MODEL_NAME == "branch_vgg16":
            model = BranchVGG16(n_classes=train_dataset.hierarchy_size, n_branches=N_BRANCHES, eps=0, lr=LEARNING_RATE)
        elif MODEL_NAME == "resnet":
            model = ResNetClassifier(num_classes=train_dataset.hierarchy_size[-1], learning_rate=LEARNING_RATE)
        elif MODEL_NAME == "hresnet":
            model = HResNet(num_classes=train_dataset.hierarchy_size, learning_rate=LEARNING_RATE)
        elif MODEL_NAME == 'condhresnet':
            model = CondHResNet(num_classes=train_dataset.hierarchy_size, learning_rate=LEARNING_RATE)


    trainer.fit(model, train_loader, val_loader)