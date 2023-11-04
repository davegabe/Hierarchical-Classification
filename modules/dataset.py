from torch.utils.data import Dataset

class HierarchicalImageNet(Dataset):
    def __init__(self):
        # Load the ImageNet dataset
        # Hint: Use torchvision.datasets.ImageNet
        pass

    def __len__(self):
        # Return the length of the dataset
        pass

    def __getitem__(self, idx):
        # Return the image and the hierarchical labels
        pass