from modules.models import VGG_19
from modules.dataset import HierarchicalImageNet
import torch

# Load the dataset
dataset = HierarchicalImageNet()

# Load the model
model = VGG_19()

# ...