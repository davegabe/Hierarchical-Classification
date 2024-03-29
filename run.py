import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from modules.models.branch_resnet import BranchResNet
from modules.models.condhresnet import CondHResNet
from modules.models.vgg11_hcnn import VGG11_HCNN
from modules.models.vgg16 import VGG16
from modules.models.vgg16_hcnn import VGG16_HCNN
from modules.models.resnet import ResNetClassifier
from modules.models.hresnet import HResNet
from modules.utils.dataset import HierarchicalImageNet
from torch.utils.data import DataLoader
from config import *
import random
import numpy as np
from pytorch_lightning.callbacks import LearningRateFinder, EarlyStopping, RichProgressBar
import nltk
import wandb


def main():
    # Print the model name
    print("#" * 80)
    print(f"Training {MODEL_NAME}")
    print("#" * 80)
    
    # Set random seeds for reproducibility
    random_state = 42
    torch.random.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    torch.set_float32_matmul_precision('medium')

    # Download the WordNet package
    nltk.download('wordnet')

    only_leaves = MODEL_NAME == 'vgg16' or MODEL_NAME == 'resnet'
    train_dataset = HierarchicalImageNet(split=TRAIN_DATASET_PATH, only_leaves=only_leaves, random_state=random_state)
    val_dataset = HierarchicalImageNet(split=VAL_DATASET_PATH, only_leaves=only_leaves, random_state=random_state)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)

    # Loggers
    loggers = [
        TensorBoardLogger("models/lightning_logs", name=MODEL_NAME)
    ]
    if USE_WANDB:
        wandb_logger = WandbLogger(project="hierarchical-classification")
        wandb_logger.experiment.config.update({
            "model": MODEL_NAME,
            "n_branches": N_BRANCHES,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "limit_classes": LIMIT_CLASSES,
            "image_size": IMAGE_SIZE,
            # "loss_weights": loss.weights if MODEL_NAME=='branch_vgg16' else None
        })
        loggers += [wandb_logger]

    trainer = L.Trainer(
        logger=loggers,
        max_epochs=NUM_EPOCHS,
        benchmark=True,
        default_root_dir='./models',
        enable_checkpointing=True,
        # precision="bf16-mixed",
        accumulate_grad_batches=4,
        callbacks=[
            LearningRateFinder(min_lr=1e-6, max_lr=1e-1),
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
        elif MODEL_NAME == "resnet":
            model = ResNetClassifier(num_classes=train_dataset.hierarchy_size[-1], learning_rate=LEARNING_RATE)
        elif MODEL_NAME == "hresnet":
            model = HResNet(num_classes=train_dataset.hierarchy_size, learning_rate=LEARNING_RATE)
        elif MODEL_NAME == 'condhresnet':
            model = CondHResNet(num_classes=train_dataset.hierarchy_size, learning_rate=LEARNING_RATE)
        elif MODEL_NAME == 'branch_resnet':
            model = BranchResNet(num_classes=train_dataset.hierarchy_size, n_branches=N_BRANCHES, learning_rate=LEARNING_RATE)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Close the loggers
    for logger in loggers:
        logger.finalize("success")
        if isinstance(logger, WandbLogger):
            wandb.finish()

    # Save the model
    trainer.save_checkpoint(f"models/{MODEL_NAME}_{NUM_EPOCHS}.ckpt")


if __name__ == "__main__":
    if MODEL_NAME == "all":
        # If MODEL_NAME is "all", train all models
        for model in ALL_MODELS:
            MODEL_NAME = model
            main()
    else:
        # Train the specified model
        main()
