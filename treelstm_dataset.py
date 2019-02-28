import os
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import Dataset

# import Constants
from tree import Tree


# Dataset class for SICK dataset
class TreeDataset(Dataset):
    def __init__(self, tree_path,ids):
        super(TreeDataset, self).__init__()

        self.trees = self.read_trees(tree_path)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        word, chars, label = self.ids[index]
        return (tree, word, label)


    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    # def read_labels(self, filename):
    #     with open(filename, 'r') as f:
    #         labels = list(map(lambda x: float(x), f.readlines()))
    #         labels = torch.Tensor(labels)
    #     return labels
