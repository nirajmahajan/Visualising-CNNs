import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image

class Digit_detection(Dataset):
    def __init__(self, train , path = '../datasets/Digit_detection', transform = None, bootstrapping = False):
        super(Digit_detection, self).__init__()
        self.train = train
        self.transform = transform
        self.bootstrapping = bootstrapping
        if self.train:
            self.datapath = os.path.join(path, 'train')
        else:
            self.datapath = os.path.join(path, 'test')

        
        self.dataset = torchvision.datasets.ImageFolder(self.datapath, transform = self.transform)
        self.labels = np.array([x[1] for x in self.dataset.samples])
        self.num_zeros = (self.labels==0).sum()
        self.num_ones = (self.labels==1).sum()
        self.generate_index_mapping()

    def generate_index_mapping(self):
        if self.bootstrapping:
            size_class = max(self.num_ones, self.num_zeros)
            self.mapping = torch.zeros(2*size_class).int()
            pointer = 0
            self.mapping[pointer:pointer+self.num_zeros] = torch.arange(self.num_zeros)
            self.mapping[pointer+self.num_zeros:size_class] = torch.tensor(np.random.choice(torch.arange(self.num_zeros), size_class-self.num_zeros)).int()
            pointer += size_class
            self.mapping[pointer:pointer+self.num_ones] = torch.arange(self.num_ones)+self.num_zeros
            self.mapping[pointer+self.num_ones:2*size_class] = torch.tensor(np.random.choice(torch.arange(self.num_ones)+self.num_zeros, size_class-self.num_ones)).int()
        else:
            self.mapping = torch.arange(self.num_ones + self.num_zeros).int()

    def __getitem__(self, i):
        index = self.mapping[i]
        return self.dataset[index]

    def __len__(self):
        return len(self.mapping)

class CIFAR10(Dataset):
    def __init__(self, train , path = '../datasets/CIFAR10', transform = None, bootstrapping = False):
        super(CIFAR10, self).__init__()
        self.train = train
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR10(root = path, download = True, transform = self.transform, train = train)

    def __getitem__(self, i):
        return self.dataset[i]

    def generate_index_mapping(self):
        pass

    def __len__(self):
        return len(self.dataset)

class CIFAR100(Dataset):
    def __init__(self, train , path = '../datasets/CIFAR100', transform = None, bootstrapping = False):
        super(CIFAR100, self).__init__()
        self.train = train
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR100(root = path, download = True, transform = self.transform, train = train)

    def __getitem__(self, i):
        return self.dataset[i]

    def generate_index_mapping(self):
        pass

    def __len__(self):
        return len(self.dataset)