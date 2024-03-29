from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn
from torchvision.transforms import v2
import pandas as pd
import os
import torch
from PIL import Image
from config import IMAGE_SIZE, LIMIT_IMAGES, LIMIT_CLASSES


class HierarchicalImageNet(Dataset):
    def __init__(self, split: str, root: str = "./dataset/", only_leaves: bool = False, random_state=42) -> None:
        self.root = root
        self.split = split
        self.only_leaves = only_leaves
        self.random_state = random_state
        self.hierarchy = self.get_hierarchy()
        self.classes, self.classes_index = self.get_classes()
        self.imagenet = self.get_imagenet()

        self.hierarchy_depth = len(self.hierarchy.columns)

        self.hierarchy_size = [
            len(self.hierarchy.iloc[:, i].unique())
            for i in range(self.hierarchy_depth)
        ]

        self.depth_class_to_index = self.get_depth_class_to_index()

    def __len__(self) -> int:
        return len(self.imagenet)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, class_name = self.imagenet[index]
        image = Image.open(image_path, mode="r").convert("RGB")
        transform = v2.Compose([
            # v2.Resize(IMAGE_SIZE),
            v2.RandomResizedCrop(size=IMAGE_SIZE, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(135),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        image = transform(image)

        class_index = self.classes_index[class_name]

        if self.only_leaves:
            return image, torch.tensor(class_index)

        class_hierarchy = self.hierarchy.iloc[class_index, :]

        hierarchy = [
            self.depth_class_to_index[i][class_hierarchy.iloc[i]]
            for i in range(self.hierarchy_depth)
        ]

        hierarchy_one_hots = [
            torch.zeros(self.hierarchy_size[i])
            for i in range(self.hierarchy_depth)
        ]

        for i, index in enumerate(hierarchy):
            hierarchy_one_hots[i][index] = 1

        hierarchy_one_hot = torch.cat(hierarchy_one_hots)

        return image, hierarchy_one_hot

    def get_depth_class_to_index(self) -> dict[int, dict[str, int]]:
        """
        For each depth create a class to index mapping.

        Returns:
            class_to_index (dict): A dictionary containing the class to index mapping
        """
        class_to_index = {}
        for depth in range(3):
            # Get the classes at the current depth
            classes = self.hierarchy.iloc[:, depth].unique()
            # Create the mapping
            class_to_index[depth] = {
                class_name: i for i, class_name in enumerate(classes)}
        return class_to_index

    def get_hierarchy(self) -> pd.DataFrame:
        # Read the hierarchy
        hierarchy = pd.read_csv("dataset/hierarchy.csv")
        # Sample random rows
        hierarchy = hierarchy.sample(n=LIMIT_CLASSES, random_state=self.random_state).reset_index(drop=True)
        return hierarchy

    def get_classes(self) -> tuple[list[str], dict[str, int]]:
        # Get 3rd column from the hierarchy
        classes = []
        classes_index = {}
        classes_names = self.hierarchy.iloc[:, 2].unique().tolist()
        print(f'Number of classes: {len(classes_names)}')
        for i, class_name in enumerate(classes_names):
            synset = wn.synset(class_name)
            pos = synset.pos()
            offset = synset.offset()
            classes.append(f"{pos}{offset:08d}")
            classes_index[f"{pos}{offset:08d}"] = i
        return classes, classes_index

    def get_imagenet(self) -> list[tuple[str, str]]:
        imagenet_path = os.path.join(self.split)
        imagenet = []
        for class_name in self.classes:
            class_path = os.path.join(imagenet_path, class_name)
            images = os.listdir(class_path)[:LIMIT_IMAGES]
            for image_name in images:
                image_path = os.path.join(class_path, image_name)
                imagenet.append((image_path, class_name))

        return imagenet
