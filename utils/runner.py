import torch.utils
import torch.utils.data
import torch
import numpy as np
import argparse
import wandb
import random
from typing import Any, Union
import torchvision
import os

from models.som import SOM
from utils.inputdata import InputData
from models.topological_AE import TopologicalAE
from utils.trainer import TopologicalAETrainer
from utils.config import Config

class Runner():

	def __init__(self, dataset_name: str):
		super().__init__()
		self.dataset_name=dataset_name

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Device: ", self.device)
		
		args = self.parse_arguments()
		self.wandb_log=args.wandb_log
		self.training_mode=args.training_mode

		assert self.training_mode in self.available_training_modes(), "Training mode should be one of "+str(self.available_training_modes())

		self.dataset_train, self.dataset_val = self.create_dataset(subset_size=None)


	def parse_arguments(self):
		"""
		Parse command line arguments.
		
		Returns:
			argparse.Namespace: Parsed command line arguments.
		"""
		parser = argparse.ArgumentParser(
						prog='TopologicalAE training',
						description='this script can train a TopologicalAE',
						epilog='Text at the bottom of help')
		parser.add_argument("--training", dest='training_mode', help="The training mode. Could be 'pytorch_batch', 'LifeLong'", type=str, required=True)
		parser.add_argument("--log", dest='wandb_log', help="Add '--log' to log in wandb.", action='store_true')
		return parser.parse_args()


	def select_training(self, model: TopologicalAE, config: Config):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		
		if self.wandb_log:
			wandb.init(project='TopologicalAE-'+config.weights_and_biases_config.PROJECT+'-'+self.dataset_name, job_type= self.training_mode)

		print("You have choose to train a TopologicalAE model with "+self.training_mode+" mode.")
		
		trainer = TopologicalAETrainer(model=model, device=self.device, wandb_log=self.wandb_log, clip_images=True)

		training_function = getattr(trainer, "train_"+self.training_mode)

		training_function(self.dataset_train, self.dataset_val, **getattr(config, self.training_mode+"_config").to_dict(), **vars(config.variables))
		return
	

	def available_training_modes(self):

			return ["pytorch_batch", "LifeLong"]
		

	def create_dataset(self, input_data: InputData=None, subset_size=None):
		
		if input_data is not None:
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]) if input_data.transform_data else torchvision.transforms.ToTensor()
		else:
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

		if self.dataset_name=="MNIST":

			self.input_data = InputData((28,28), channels=1, channel_range="RGB")
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]) if self.input_data.transform_data else torchvision.transforms.ToTensor()
			# data in .data and labels in .targets
			MNIST_train = torchvision.datasets.MNIST(
				root=os.path.curdir,
				train=True,
				download=True,
				transform=transform
			)
			MNIST_val = torchvision.datasets.MNIST(
				root=os.path.curdir,
				train=False,
				download=True,
				transform=transform
			)	

			if subset_size is not None:
				MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(subset_size)])
				MNIST_train_subset.targets=MNIST_train.targets[0:subset_size]
				MNIST_val_subset= torch.utils.data.dataset.Subset(MNIST_val,[i for i in range(subset_size)])
				MNIST_val_subset.targets=MNIST_val.targets[0:subset_size]
				return MNIST_train_subset, MNIST_val_subset
			return MNIST_train, MNIST_val


		
		if self.dataset_name=="CIFAR10":
			self.input_data = InputData((32,32), channels=3, channel_range="RGB")
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]) if self.input_data.transform_data else torchvision.transforms.ToTensor()
			# data in .data and labels in .targets
			CIFAR_train = torchvision.datasets.CIFAR10(
				root=os.path.curdir,
				train=True,
				download=True,
				transform=transform
			)
			CIFAR_val = torchvision.datasets.CIFAR10(
				root=os.path.curdir,
				train=False,
				download=True,
				transform=transform
			)
			
			if subset_size is not None:
				CIFAR_train_subset= torch.utils.data.dataset.Subset(CIFAR_train,[i for i in range(subset_size)])
				CIFAR_train_subset.targets=torch.Tensor(CIFAR_train.targets[0:subset_size])
				CIFAR_val_subset= torch.utils.data.dataset.Subset(CIFAR_val,[i for i in range(subset_size)])
				CIFAR_val_subset.targets=torch.Tensor(CIFAR_val.targets[0:subset_size])	
				return CIFAR_train_subset, CIFAR_val_subset
			return CIFAR_train, CIFAR_val

		if self.dataset_name=="FASHION-MNIST":
			self.input_data = InputData((28,28), channels=1, channel_range="RGB")
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]) if self.input_data.transform_data else torchvision.transforms.ToTensor()
			# data in .data and labels in .targets
			FASHION_train = torchvision.datasets.FashionMNIST(
				root=os.path.curdir,
				train=True,
				download=True,
				transform=transform
			)
			FASHION_val = torchvision.datasets.FashionMNIST(
				root=os.path.curdir,
				train=False,
				download=True,
				transform=transform
			)
			if subset_size is not None:
				FASHION_train_subset= torch.utils.data.dataset.Subset(FASHION_train,[i for i in range(subset_size)])
				FASHION_train_subset.targets=torch.Tensor(FASHION_train.targets[0:subset_size])
				FASHION_val_subset= torch.utils.data.dataset.Subset(FASHION_val,[i for i in range(subset_size)])
				FASHION_val_subset.targets=torch.Tensor(FASHION_val.targets[0:subset_size])	
				return FASHION_train_subset, FASHION_val_subset
			return FASHION_train, FASHION_val


	def run(self, config: Config):
		"""
		Main function to run the training and plotting of the SOM/STM.
		"""
		torch.manual_seed(config.variables.SEED)
		random.seed(config.variables.SEED)
		np.random.seed(config.variables.SEED)
		
		model = TopologicalAE(config.tae, config.variables.SEED).to(self.device)

		self.select_training(model, config)

