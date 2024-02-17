# Dataset parameters
IMAGE_SIZE = (128, 128)

# Training parameters
MODEL_NAME = "vgg16"  # "vgg16" or "branch_vgg16"
BRANCH_SELECTOR = "learnable"  # "static" or "learnable"
N_BRANCHES = 3
LEARNING_RATE = 3.311311214825912e-05
L1_REGULARIZATION = 1e-5
LOG_STEP = 1
PRIVILEGED = False
SIMILARITY_REGULARIZATION = 0.001
NUM_EPOCHS = 50
VAL_EPOCHS = 5
BATCH_SIZE = 24
LIMIT_CLASSES = 100 # -1 means no limit, use all classes
LIMIT_IMAGES = 500 # limit the number of images per class, -1 means no limit

TRAIN_DATASET_PATH = "dataset/train"
VAL_DATASET_PATH = "dataset/val"

USE_WANDB = False