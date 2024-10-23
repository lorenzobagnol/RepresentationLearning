import pickle
import os
import numpy as np
import torch
import random
import torchvision

from utils.inputdata import InputData
from utils.runner import BaseRunner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig





class CifarRunner(BaseRunner):

	def __init__(self, config: Config, dataset_name: str, input_data: InputData):
		super().__init__(config=config, dataset_name=dataset_name, input_data=input_data)
		
	def create_dataset(self):
		"""
		"""
		# data in .data and labels in .targets
		CIFAR_train = torchvision.datasets.CIFAR10(
			root=os.path.curdir,
			train=True,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		CIFAR_val = torchvision.datasets.CIFAR10(
			root=os.path.curdir,
			train=False,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		CIFAR_train_subset= torch.utils.data.dataset.Subset(CIFAR_train,[i for i in range(10000)])
		CIFAR_train_subset.targets=torch.Tensor(CIFAR_train.targets[0:10000])
		CIFAR_val_subset= torch.utils.data.dataset.Subset(CIFAR_val,[i for i in range(10000)])
		CIFAR_val_subset.targets=torch.Tensor(CIFAR_val.targets[0:10000])	
	
		# target_points=self.generate_equally_distributed_points(10)
		points = [
            [ 3,  3],
			[ 2, 10],
			[ 3, 16],
			[10,  3],
			[ 7,  9],
			[12, 10],
			[ 9, 16],
			[16,  3],
			[17, 10],
			[16, 16]
        ]
		random.shuffle(points)
		target_points={k : torch.Tensor(v) for k,v in enumerate(points)}

		return CIFAR_train_subset, CIFAR_val_subset, target_points



input_data=InputData((32,32),3,"Unit")
config = Config(
	SEED=13,
    som_config=SOMConfig(M=20, N=20, SIGMA=10),
    lifelong_config=LifeLongConfig(ALPHA=10, BETA=0.01, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
    simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20),
    pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001),
    online_config=OnlineConfig(EPOCHS=1)
)
random.seed(config.SEED)
cifar_runner=CifarRunner(config=config, dataset_name="cifar", input_data=input_data)
cifar_runner.run()


