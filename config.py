# Dataset parameters
IMAGE_SIZE = (224, 224)

# Training parameters
MODEL_NAME = "vgg16"  # "vgg16" or "branch_vgg16"
N_BRANCHES = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 24
LIMIT_CLASSES = 10 # -1 means no limit, use all classes