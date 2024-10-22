from typing import Sequence, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import types
import torchvision


# data in .data and labels in .targets
train_MNIST_dataset = torchvision.datasets.MNIST(
    root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)

MNIST_dataset = torchvision.datasets.MNIST(
    root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
)


torch.stack([MNIST_dataset.data[0]]).to(torch.float)