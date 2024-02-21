# Training parameters
MODEL_NAME = "branch_resnet"  # "vgg16" or "vgg16_hcnn" or "vgg11_hcnn" or "resnet" or "hresnet" or "condhresnet" or "branch_resnet"
N_BRANCHES = 3
LEARNING_RATE = 3.311311214825912e-05
PRIVILEGED = True
NUM_EPOCHS = 50
USE_WANDB = True
BATCH_SIZE = 64

# Dataset parameters
IMAGE_SIZE = (128, 128)
LIMIT_CLASSES = 100 # -1 means no limit, use all classes
LIMIT_IMAGES = 500 # limit the number of images per class, -1 means no limit
TRAIN_DATASET_PATH = "dataset/train"
VAL_DATASET_PATH = "dataset/val"