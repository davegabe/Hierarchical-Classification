from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn
import nltk
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
import os
import torch

nltk.download('wordnet')


class HierarchicalImageNet(Dataset):
    def __init__(self, split: str, root: str = "./dataset/"):
        self.root = root
        self.split = split
        self.max_hierarchy_depth = 7

        # Load the ImageNet dataset
        self.imagenet, self.classes = self.get_imagenet()
        self.n_classes = len(self.classes)
        print(f"Number of classes: {self.n_classes}")
        # Load the hierarchy
        self.hierarchy = self.get_hierarchy()
        print(self.hierarchy.iloc[:, 0].value_counts())

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.imagenet[idx]
        class_name = image.split('_')[0]
        class_id = self.classes.index(class_name)
        # Load the image
        image_path = os.path.join(self.root, self.split, class_name, image)
        sample = transforms.ToTensor()(image_path)
        # Get the hierarchy of the class
        hierarchy = self.hierarchy.iloc[class_id, :]
        # TODO: Convert the hierarchy to one-hot encoding
        hierarchy_one_hot = torch.tensor(hierarchy)
        return sample, hierarchy_one_hot

    def get_imagenet(self) -> tuple[list[str], list[str]]:
        """
        Returns the ImageNet dataset as a dictionary

        Args:
            root (str): The root directory of the dataset

        Returns:
            imagenet (dict): A dictionary containing the ImageNet dataset
        """
        # Load the ImageNet dataset
        split_path = os.path.join(self.root, self.split)
        classes = []
        imagenet = []
        # For each class
        for class_name in os.listdir(split_path):
            classes.append(class_name)
            # Add the path of each image to the list
            images = os.listdir(os.path.join(split_path, class_name))
            imagenet += images
        return imagenet, classes

    def get_hierarchy(self) -> pd.DataFrame:
        """
        Returns the hierarchy of the dataset

        Returns:
            hierarchy (dict): A dictionary containing the hierarchy of the dataset
        """
        # Read synset ids for classes
        map_fname = os.path.join(self.root, "LOC_synset_mapping.txt")
        synset_ids = []
        with open(map_fname, 'r') as f:
            for line in f:
                synset_id_s = line.split()[0]  # e.g. n01440764
                if synset_id_s not in self.classes:
                    continue
                synset_id = int(synset_id_s[1:])  # e.g. 01440764
                synset_ids.append(synset_id)
        synset_ids = np.array(synset_ids)

        # Create the list of parents for all classes
        all_list = [
            ["-" for j in range(self.max_hierarchy_depth)]
            for i in range(self.n_classes)
        ]
        for i in range(self.n_classes):
            synset = wn.synset_from_pos_and_offset('n', int(synset_ids[i]))
            hyper_list = [synset.name()]
            while synset.hypernyms():
                synset = synset.hypernyms()[0]
                hyper_list.append(synset.name())
            hyper_list.insert(0, 'null')
            hyper_list.insert(0, 'null')
            all_list[i][:] = hyper_list[self.max_hierarchy_depth:-1]

        # Create the dataframe
        df = pd.DataFrame(all_list)
        return df
