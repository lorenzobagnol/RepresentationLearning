import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import Tensor
import torchvision
import os
import wandb

from models.stm import STM
from utils.trainer import SOMTrainer
from utils.inputdata import InputData

def create_dataset(input_data: InputData):
	"""
	"""
	# data in .data and labels in .targets
	MNIST_train = torchvision.datasets.MNIST(
		root=os.path.curdir,
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	MNIST_val = torchvision.datasets.MNIST(
		root=os.path.curdir,
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(10000)])
	MNIST_train_subset.targets=MNIST_train.targets[0:10000]
	indices_one_label = torch.where(MNIST_train_subset.targets==0)[0].tolist()
	MNIST_train_one_label=torch.utils.data.Subset(MNIST_train_subset, indices_one_label)
	MNIST_train_one_label.targets=MNIST_train_subset.targets[indices_one_label]


	MNIST_val_subset= torch.utils.data.dataset.Subset(MNIST_val,[i for i in range(10000)])
	MNIST_val_subset.targets=MNIST_val.targets[0:10000]
	indices_one_label = torch.where(MNIST_val_subset.targets==0)[0].tolist()
	MNIST_val_one_label=torch.utils.data.Subset(MNIST_val_subset, indices_one_label)
	MNIST_val_one_label.targets=MNIST_val_subset.targets[indices_one_label]

	return MNIST_train_one_label, MNIST_val_one_label

def generate_equally_distributed_points_v2(m, n, device, P: int=None) -> dict[int, Tensor]:
	points = np.array(
			[
				[0.15, 0.17],
				[0.12, 0.54],
				[0.16, 0.84],
				[0.50, 0.15],
				[0.36, 0.45],
				[0.62, 0.50],
				[0.48, 0.82],
				[0.83, 0.17],
				[0.88, 0.50],
				[0.83, 0.83],
			]
		)
	points_list=np.int32(points*min(m, n)).tolist()
	random.seed(13)
	random.shuffle(points_list)
	dict_points={k : torch.Tensor(v).to(device) for k,v in enumerate(points_list)}
	return dict_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_data = InputData((28, 28), 1, "Unit")
target_points=generate_equally_distributed_points_v2(20,20,device)
stm = STM(m=20, n=20, sigma=10, input_data=input_data, target_points=target_points).to(device)
dataset_train, dataset_val = create_dataset(input_data=input_data)


trainer = SOMTrainer(stm, True, True)
wandb.init(project='STM-one-label-MNIST')
trainer.train_pytorch_batch(
	dataset_train=dataset_train,
	dataset_val=dataset_val,
	EPOCHS=200, 
	BATCH_SIZE=20, 
	LEARNING_RATE=1, 
	BETA=0.02,
	target_radius=2
	)
