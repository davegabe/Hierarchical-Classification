ALL_MODELS = ["vgg11_hcnn", "vgg16", "vgg16_hcnn", "resnet", "hresnet", "condhresnet", "branch_resnet"]

# Training parameters
MODEL_NAME = "all"  # "vgg11_hcnn" or "vgg16" or "resnet" or "hresnet" or "condhresnet" or "branch_resnet" or "all"
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