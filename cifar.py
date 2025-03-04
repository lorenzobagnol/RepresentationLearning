import pickle
import os
import numpy as np
import torch
import random
import torchvision

from utils.inputdata import InputData
from utils.runner import Runner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig



def create_dataset():
	"""
	"""
	# data in .data and labels in .targets
	CIFAR_train = torchvision.datasets.CIFAR10(
		root=os.path.curdir,
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	CIFAR_val = torchvision.datasets.CIFAR10(
		root=os.path.curdir,
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	CIFAR_train_subset= torch.utils.data.dataset.Subset(CIFAR_train,[i for i in range(10000)])
	CIFAR_train_subset.targets=torch.Tensor(CIFAR_train.targets[0:10000])
	CIFAR_val_subset= torch.utils.data.dataset.Subset(CIFAR_val,[i for i in range(10000)])
	CIFAR_val_subset.targets=torch.Tensor(CIFAR_val.targets[0:10000])	

	return CIFAR_train_subset, CIFAR_val_subset



input_data=InputData((32,32),3,"RGB")
config = Config(
	SEED=13,
    som_config=SOMConfig(M=20, N=20, SIGMA=10),
    lifelong_config=LifeLongConfig(ALPHA=10, BETA=0.01, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
    simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
    pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
    online_config=OnlineConfig(EPOCHS=1)
)
random.seed(config.SEED)
dataset_train, dataset_val = create_dataset(input_data=input_data)
cifar_runner=Runner(config=config, dataset_name="CIFAR", input_data=input_data, train_dataset=dataset_train, val_dataset=dataset_val)
cifar_runner.run()



