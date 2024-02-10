# Dataset parameters
IMAGE_SIZE = (224, 224)

# Training parameters
MODEL_NAME = "branch_vgg16"  # "vgg16" or "branch_vgg16"
N_BRANCHES = 3
LEARNING_RATE = 0.0001
NUM_EPOCHS = 500
BATCH_SIZE = 17
LIMIT_CLASSES = 10 # -1 means no limit, use all classes