from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus.reader.wordnet import Synset
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from config import *

nltk.download('wordnet')


class HierarchicalImageNet(Dataset):
    def __init__(self, split: str, root: str = "./dataset/"):
        self.root = root
        self.split = split
        self.max_hierarchy_depth = None
        self.desired_hierarchy_depth = 4

        # Load the ImageNet dataset
        self.imagenet, self.classes = self.get_imagenet()
        self.n_classes = len(self.classes)
        print(f"Number of classes: {self.n_classes}")
        # Load the hierarchy
        self.hierarchy = self.get_hierarchy()

        # Dict class -> index for each depth
        self.depth_class_to_index = self.get_depth_class_to_index()

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.imagenet[idx]
        class_name = image.split('_')[0]
        class_id = self.classes.index(class_name)
        # Load the image
        image_path = os.path.join(self.root, self.split, class_name, image)
        image = Image.open(image_path)
        # Transform the image
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        sample = transform(image)
        # Get the hierarchy of the class
        hierarchy = self.hierarchy.iloc[class_id, :]
        hierarchy = [
            self.depth_class_to_index[i][hierarchy[i]]
            for i in range(self.desired_hierarchy_depth)
        ]
        hierarchy_one_hot = torch.tensor(hierarchy)
        return sample, hierarchy_one_hot

    def get_depth_class_to_index(self) -> dict[int, dict[str, int]]:
        """
        For each depth create a class to index mapping.

        Returns:
            class_to_index (dict): A dictionary containing the class to index mapping
        """
        class_to_index = {}
        for depth in range(self.desired_hierarchy_depth):
            # Get the classes at the current depth
            classes = self.hierarchy.iloc[:, depth].unique()
            # Create the mapping
            class_to_index[depth] = {
                class_name: i for i, class_name in enumerate(classes)}
        return class_to_index

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

    def get_max_depth(self, root, nodes, synsets) -> int:
        """
        Returns the maximum depth of the hierarchy

        Returns:
            max_depth (int): The maximum depth of the hierarchy
        """
        max_depth = 0
        for synset in synsets:
            depth = 0
            name = synset.name()
            while name != root:
                name = nodes[name]['parent']
                depth += 1
            if depth > max_depth:
                max_depth = depth
        return max_depth

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

        # Find the root of the tree
        root = None
        for name, node in nodes.items():
            if node['parent'] is None:
                root = name
                break
        # # Find the first node with multiple children (common root)
        # while len(nodes[root]['children']) == 1:
        #     root = nodes[root]['children'][0]

        self.max_hierarchy_depth = self.get_max_depth(root, nodes, synsets)
        print(f"Max depth: {self.max_hierarchy_depth}")

        # Collapse nodes with #children == min_num_children until the desired depth is reached
        while self.max_hierarchy_depth > self.desired_hierarchy_depth:
            nodes_copy = nodes.copy()
            min_children = self.min_number_of_children(nodes, synsets)
            for name, node in nodes_copy.items():
                if len(node['children']) == min_children and len(node['children']) != 0:
                    # Update the children of the parent
                    if node['parent'] is None:
                        continue
                    parent = node['parent']
                    children = node['children']
                    nodes[parent]['children'].remove(name)
                    nodes[parent]['children'] += children
                    # Update the parent of the children
                    for child in children:
                        nodes[child]['parent'] = parent
                    # Delete the node
                    del nodes[name]
            self.max_hierarchy_depth = self.get_max_depth(root, nodes, synsets)

        # Create the hierarchy dataframe starting from the root
        all_list = [[] for _ in range(self.n_classes)]
        for i, synset in enumerate(synsets):
            # Get the hierarchy of the synset
            name = synset.name()
            depth = 0
            while name != root:
                all_list[i].append(name)
                name = nodes[name]['parent']
                depth += 1
            # Add "-" to the hierarchy to make it of the same length
            for j in range(depth, self.max_hierarchy_depth):
                all_list[i].insert(0, "-")
            # Reverse the hierarchy
            all_list[i].reverse()

        # Create the dataframe
        df = pd.DataFrame(all_list)
        return df

    def min_number_of_children(self, nodes, synsets) -> int:
        """
        Returns the minimum number of children of a node in the hierarchy, skips nodes with no children

        Returns:
            min_num_children (int): The minimum number of children of a node in the hierarchy
        """
        childrened_nodes = [name for name,
                            node in nodes.items() if len(node['children']) != 0]
        min_num_children = np.inf
        for name in childrened_nodes:
            if nodes[name]['parent'] is None:
                continue
            num_children = len(nodes[name]['children'])
            if num_children < min_num_children:
                min_num_children = num_children
        return min_num_children


if __name__ == "__main__":
    # Load the dataset
    dataset = HierarchicalImageNet("train")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Print dataset info
    depth_class_to_index = dataset.depth_class_to_index
    # Format the dictionary on multiple lines
    depth_class_to_index_str = "{\n"
    for depth, class_to_index in depth_class_to_index.items():
        depth_class_to_index_str += f"    {depth}: {class_to_index},\n\n"
    depth_class_to_index_str += "}"
    print(f"Depth class to index: {depth_class_to_index_str}")
    
    # Load one batch and show in matplotlib
    images, hierarchies = next(iter(dataloader))
    # For each sample in the batch
    for i in range(BATCH_SIZE):
        # Get the class name
        hierarchy = hierarchies[i]
        # Get the image
        image = images[i].permute(1, 2, 0)
        # Show the image
        plt.imshow(image)
        plt.title(hierarchy)
        plt.show()
