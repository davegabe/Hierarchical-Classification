# Dataset parameters
IMAGE_SIZE = (224, 224)

# Training parameters
MODEL_NAME = "branch_vgg16"  # "vgg16" or "branch_vgg16"
BRANCH_SELECTOR = "learnable"  # "static" or "learnable"
N_BRANCHES = 3
LEARNING_RATE = 0.0001
L1_REGULARIZATION = 1
SIMILARITY_REGULARIZATION = 0.1
NUM_EPOCHS = 200
BATCH_SIZE = 64
LIMIT_CLASSES = -1 # -1 means no limit, use all classes
LIMIT_IMAGES = 100 # limit the number of images per class, -1 means no limit