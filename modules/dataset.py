from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus.reader.wordnet import Synset
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
        synsets = []
        with open(map_fname, 'r') as f:
            for line in f:
                synset_id_s = line.split()[0]  # e.g. n01440764
                if synset_id_s not in self.classes:
                    continue
                synset_id = int(synset_id_s[1:])  # e.g. 01440764
                synset: Synset = wn.synset_from_pos_and_offset('n', synset_id)
                synsets.append(synset)

        # Initialize the nodes
        nodes = {}
        for synset in synsets:
            # Get the hypernyms of the synset
            nodes[synset.name()] = {'children': [], 'parent': None}

        # Create the tree from the synset ids
        for synset in synsets:
            # For each hypernym of the synset (and for each hypernym of hypernyms)
            while synset.hypernyms():
                # Get the hypernyms of the synset
                hypernym = synset.hypernyms()[0]
                name = hypernym.name()
                nodes[synset.name()]['parent'] = name
                if name not in nodes:
                    nodes[name] = {'children': [synset.name()], 'parent': None}
                if synset.name() not in nodes[name]['children']:
                    nodes[name]['children'].append(synset.name())
                # Set the synset to the hypernym
                synset = hypernym

        # Collapse nodes with only one child into their child
        while True:
            changed = False
            # For each leaf node
            for synset in synsets:
                parent = nodes[synset.name()]['parent']
                if parent is None:
                    raise ValueError(f"Node {synset.name()} has no parent")
                # If the node has only one child
                if len(nodes[parent]['children']) == 1:
                    # The only child of the parent is synset.name()
                    # Remove the parent from the tree
                    grandparent = nodes[parent]['parent']
                    nodes[grandparent]['children'].remove(parent)
                    nodes[grandparent]['children'].append(synset.name())
                    nodes[synset.name()]['parent'] = grandparent
                    # Remove the parent from the list of nodes
                    del nodes[parent]
                    changed = True
            if not changed:
                break

        # Find the root of the tree
        root = None
        for name, node in nodes.items():
            if node['parent'] is None:
                root = name
                break
        # Find the first node with multiple children (common root)
        while len(nodes[root]['children']) == 1:
            root = nodes[root]['children'][0]

        # Compute the depth of each node
        max_depth = 0
        for synset in synsets:
            depth = 0
            name = synset.name()
            while name != root:
                name = nodes[name]['parent']
                depth += 1
            if depth > max_depth:
                max_depth = depth
        print(f"Max depth: {max_depth}")

        # Create the hierarchy dataframe starting from the root
        all_list = []
        # TODO: Create the hierarchy dataframe starting from the root

        # Create the dataframe
        df = pd.DataFrame(all_list)
        return df
