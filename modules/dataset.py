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
    def __init__(self, split: str, root: str = "./dataset/", only_leaves: bool = False):
        self.root = root
        self.split = split
        self.max_hierarchy_depth = None
        self.desired_hierarchy_depth = 5
        self.real_hierarchy_depth = self.desired_hierarchy_depth - 2
        self.only_leaves = False
        self.whitelist = []
        # Load the ImageNet dataset
        self.imagenet, self.classes = self.get_imagenet()
        self.n_classes = len(self.classes)
        print(f"Number of classes: {self.n_classes}")
        
        # Load the hierarchy
        self.hierarchy = self.get_hierarchy()
        # Filter the ImageNet dataset
        self.get_classes()
        self.filter_imagenet()
        

        # save csv
        # sort by alphabetical order
        self.hierarchy = self.hierarchy.sort_values(by=[0, 1, 2, 3, 4])
        # filter out entries that have '-' in the second and third column
        self.hierarchy = self.hierarchy[~self.hierarchy[1].str.contains('-')]
        self.hierarchy = self.hierarchy[~self.hierarchy[2].str.contains('-')]
        # remove column 3,4
        self.hierarchy = self.hierarchy.drop([3, 4], axis=1)
        # remove duplicates
        self.hierarchy = self.hierarchy.drop_duplicates()
        #self.hierarchy.to_csv(f"{self.split}_hierarchy.csv", index=False, header=False)
        # Dict class -> index for each depth

        # Get the number of classes for each depth
        self.hierarchy_size = [
            len(self.hierarchy.iloc[:, i].unique())
            for i in range(self.real_hierarchy_depth)
        ]

        self.depth_class_to_index = self.get_depth_class_to_index()

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, class_name = self.imagenet[idx]
        class_id = self.classes.index(class_name)
        # Load the image
        image_path = os.path.join(self.root, self.split, class_name, image)
        image = Image.open(image_path, mode="r").convert("RGB")
        # Transform the image
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        sample = transform(image)

        # Return only the leaf class if only_leaves is True
        if self.only_leaves:
            hierarchy_one_hot = torch.zeros(self.n_classes)
            hierarchy_one_hot[class_id] = 1
            return sample, hierarchy_one_hot

        hierarchy = self.hierarchy.iloc[class_id, :]
        # Get the hierarchy index of the class
        hierarchy = [
            self.depth_class_to_index[i][hierarchy[i]]
            for i in range(self.real_hierarchy_depth)
        ]

        # Create the one hot encoding of the hierarchy
        hierarchy_one_hot = [
            torch.zeros(self.hierarchy_size[i])
            for i in range(self.real_hierarchy_depth)
        ]
        # Set the one hot of the class to 1
        for i, class_id in enumerate(hierarchy):
            hierarchy_one_hot[i][class_id] = 1

        # Concatenate the one hot encodings
        hierarchy_one_hot = torch.cat(hierarchy_one_hot, dim=0)
        return sample, hierarchy_one_hot

    def get_depth_class_to_index(self) -> dict[int, dict[str, int]]:
        """
        For each depth create a class to index mapping.

        Returns:
            class_to_index (dict): A dictionary containing the class to index mapping
        """
        class_to_index = {}
        for depth in range(self.real_hierarchy_depth):
            # Get the classes at the current depth
            classes = self.hierarchy.iloc[:, depth].unique()
            # Create the mapping
            class_to_index[depth] = {
                class_name: i for i, class_name in enumerate(classes)}
        return class_to_index

    def get_imagenet(self) -> tuple[list[tuple[str, str]], list[str]]:
        """
        Returns the ImageNet dataset as a dictionary

        Args:
            root (str): The root directory of the dataset

        Returns:
            imagenet (list[tuple[str, str]]): A list containing the images and their class
            classes (list[str]): A list containing the classes of the dataset
        """
        # Load the ImageNet dataset
        split_path = os.path.join(self.root, self.split)
        imagenet = []
        classes = os.listdir(split_path)
        # For each class
        for class_name in classes:
            # Add the path of each image to the list
            path = os.path.join(split_path, class_name)
            images = os.listdir(path)
            imagenet += [(image, class_name) for image in images]
        return imagenet, classes

    def get_classes(self):
        """
        Create a list containing the classes of the dataset based on the hierarchy.
        """
        classes = []
        name_classes = self.hierarchy.iloc[:, self.real_hierarchy_depth-1].unique()
        if LIMIT_CLASSES > 0:
            name_classes = name_classes[:LIMIT_CLASSES]
        for i, class_name in enumerate(name_classes):
            if class_name == "-":
                continue
            synset = wn.synset(class_name)
            pos = synset.pos() # n
            offset = synset.offset() # 8 digits
            classes.append(f"{pos}{offset:08d}")
        self.classes = classes

    def filter_imagenet(self):
        """
        Remove the images from self.imagenet not in self.classes
        """
        new_imagenet = []
        for i, (image, class_name) in enumerate(self.imagenet):
            if class_name in self.classes:
                new_imagenet.append((image, class_name))
        self.imagenet = new_imagenet

    def get_max_depth(self, root: str, nodes: dict, synsets: list) -> int:
        """
        Returns the maximum depth of the hierarchy

        Args:
            root (str): The name of the root of the hierarchy
            nodes (dict): A dictionary containing the nodes of the hierarchy
            synsets (list): A list containing the synsets of the dataset

        Returns:
            max_depth (int): The maximum depth of the hierarchy
        """
        max_depth = 0
        for synset in synsets:
            depth = 0
            name = synset.name()
            found_root = False
            # Get the parent of the synset
            while name != None:
                # If I found the root
                if name == root:
                    found_root = True
                    break
                name = nodes[name]['parent']
                depth += 1
            # If I found the root and the depth is greater than the current max depth
            if depth > max_depth and found_root:
                max_depth = depth
        return max_depth

    def get_hierarchy(self) -> pd.DataFrame:
        """
        Returns the hierarchy of the dataset

        Returns:
            hierarchy (dict): A dictionary containing the hierarchy of the dataset
        """
        # Read synset ids for classes
        synsets = []
        for synset_id_s in self.classes:
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

        # ab_children = nodes['physical_entity.n.01']['children']
        # nodes['entity.n.01']['children'] += ab_children
        # for child in ab_children:
        #     nodes[child]['parent'] = 'entity.n.01'
        # del nodes['physical_entity.n.01']

        # Find the root of the tree
        root = None
        for name, node in nodes.items():
            if node['parent'] is None:
                root = name
                print("test", name)
        # # Find the first node with multiple children (common root)
        # while len(nodes[root]['children']) == 1:
        #     root = nodes[root]['children'][0]

        # Remove first three levels from root
        with open('blacklist.txt', 'r') as f:
            blacklist = f.read().splitlines()

        with open('whitelist.txt', 'r') as f:
            self.whitelist = f.read().splitlines()

        for child in blacklist:
            self.prune_node(child, nodes)
            if wn.synset(child) in synsets:
                parent = wn.synset(child).hypernyms()[0].name()
                synsets.remove(wn.synset(child))
                synsets.append(wn.synset(parent))





        # Collapse nodes with #children == min_num_children until the desired depth is reached
        # num_roots = 0
        # while self.max_hierarchy_depth > self.desired_hierarchy_depth:
        #     nodes_copy = nodes.copy()
        #     min_children = self.min_number_of_children(nodes, synsets)
        #     for name, node in nodes_copy.items():
        #         if len(node['children']) == min_children and len(node['children']) != 0:
        #             # if self.get_max_depth(name, nodes, synsets) <= self.desired_hierarchy_depth:
        #             #     continue
        #             if node['parent'] == None:
        #                 num_roots += 1
        #                 continue
        #             parent = node['parent']
        #             children = node['children']
        #             nodes[parent]['children'].remove(name)
        #             nodes[parent]['children'] += children
        #             # Update the parent of the children
        #             for child in children:
        #                 nodes[child]['parent'] = parent
        #             # Delete the node
        #             del nodes[name]
        #     self.max_hierarchy_depth = self.get_max_depth(root, nodes, synsets)
        # print(f"Number of roots: {num_roots}")

        leaf_nodes = [name for name, node in nodes.items()
                      if len(node['children']) == 0]
        
        # For each leaf node, prune the tree until the desired depth is reached
        for leaf_node in leaf_nodes:
            # Get depth of the leaf node
            depth = self.get_max_depth(root, nodes, synsets)
            # While the depth is greater than the desired depth
            while depth > self.desired_hierarchy_depth:
                # Get the node with the least number of children
                min_children, min_node, depth = self.get_min_children_branch(leaf_node, nodes)
                if min_node == 'artifact.n.01':
                    print("test")
                # Prune the node
                if min_node != None:
                    self.prune_node(min_node, nodes)

        depth = self.get_max_depth(root, nodes, synsets)
        
        print(nodes['entity.n.01'])
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
            for j in range(depth, self.desired_hierarchy_depth):
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

    # Removes node and merges its children with its parent
    def prune_node(self, node_name:str, nodes:dict):
        # Get the parent of the node
        if node_name == None:
            return
        parent = nodes[node_name]['parent']
        # Get the children of the node
        children = nodes[node_name]['children']

        if parent != None:
            # Remove the node from the children of the parent
            nodes[parent]['children'].remove(node_name)
            # Add the children of the node to the children of the parent
            nodes[parent]['children'] += children
        # Update the parent of the children
        for child in children:
            nodes[child]['parent'] = parent
        # Delete the node
        del nodes[node_name]

    def get_min_children_branch(self, leaf : str, nodes : dict) -> tuple[int, str]:
        min_children = np.inf
        node_name = leaf
        min_name = None
        depth = 0
        while nodes[nodes[node_name]['parent']]['parent']!= 'entity.n.01':
            parent = nodes[node_name]['parent']
            if len(nodes[parent]['children']) < min_children and parent not in self.whitelist:
                min_children = len(nodes[parent]['children'])
                min_name = parent
            node_name = parent
            depth += 1
        return min_children, min_name, depth

if __name__ == "__main__":
    # Load the dataset
    dataset = HierarchicalImageNet("train")
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # # Print dataset info
    # depth_class_to_index = dataset.depth_class_to_index
    # # Format the dictionary on multiple lines
    # depth_class_to_index_str = "{\n"
    # for depth, class_to_index in depth_class_to_index.items():
    #     depth_class_to_index_str += f"    {depth}: {class_to_index},\n\n"
    # depth_class_to_index_str += "}"
    # print(f"Depth class to index: {depth_class_to_index_str}")
    
    # # Load one batch and show in matplotlib
    # images, hierarchies = next(iter(dataloader))
    # # For each sample in the batch
    # for i in range(BATCH_SIZE):
    #     # Get the class name
    #     hierarchy = hierarchies[i]
    #     # Get the image
    #     image = images[i].permute(1, 2, 0)
    #     # Show the image
    #     plt.imshow(image)
    #     plt.title(hierarchy)
    #     plt.show()
