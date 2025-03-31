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
from utils.trainer import SOMTrainer, STMTrainer, TopologicalAETrainer
from utils.config import Config

class Runner():

	def __init__(self, config: Config, dataset_name: str):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Device: ", self.device)
		
		args = self.parse_arguments()
		self.wandb_log=args.wandb_log
		self.training_mode=args.training_mode
		self.model_name=args.model

		assert self.model_name in ["SOM", "STM", "TopologicalAE"], "Model name should be one of 'SOM', 'STM' or 'TopologicalAE'"
		assert self.training_mode in self.available_training_modes(), "Training mode should be one of "+str(self.available_training_modes())


		if self.model_name=="TopologicalAE":
			self.model = TopologicalAE(10*10).to(self.device)
			self.dataset_train, self.dataset_val = self.create_dataset()
		else:
			self.model = SOM(self.config.som_config.M, self.config.som_config.N, self.config.som_config.INPUT_DATA).to(self.device)
			self.dataset_train, self.dataset_val = self.create_dataset(self.config.som_config.INPUT_DATA)


	def parse_arguments(self):
		"""
		Parse command line arguments.
		
		Returns:
			argparse.Namespace: Parsed command line arguments.
		"""
		parser = argparse.ArgumentParser(
						prog='SOM/STM training',
						description='this script can train a SOM or a STM',
						epilog='Text at the bottom of help')
		parser.add_argument("--model", dest='model', help="The model to run. Could be 'SOM', 'STM' or 'TopologicalAE'", type=str, required=True)
		parser.add_argument("--training", dest='training_mode', help="The training mode. Could be 'simple_batch', 'online', 'pytorch_batch', 'LifeLong'", type=str, required=True)
		parser.add_argument("--log", dest='wandb_log', help="Add '--log' to log in wandb.", action='store_true')
		return parser.parse_args()


	def select_training(self, model: Union[SOM, TopologicalAE]):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		
		if self.wandb_log:
			wandb.init(project=self.model_name+'-'+self.config.weights_and_biases_config.PROJECT+'-'+self.dataset_name, job_type= self.training_mode)

		print("You have choose to train a "+self.model_name+" model with "+self.training_mode+" mode.")
		
		match self.model_name:
			case "SOM":
				trainer = SOMTrainer(model=model, device=self.device, wandb_log=self.wandb_log, clip_images=True)
			case "STM":
				trainer = STMTrainer(model=model, device=self.device, wandb_log=self.wandb_log, clip_images=True)
			case "TopologicalAE":
				trainer = TopologicalAETrainer(model=model, device=self.device, wandb_log=self.wandb_log, clip_images=True)

		training_function = getattr(trainer, "train_"+self.training_mode)

		training_function(self.dataset_train, self.dataset_val, **getattr(self.config, self.training_mode+"_config").to_dict(), **vars(self.config.variables), **self.config.som_config.to_dict())
		return
	

	def available_training_modes(self):

		if self.model_name == "STM":
			return ["LifeLong", "pytorch_batch"]
		
		if self.model_name == "SOM":
			return ["simple_batch", "online", "pytorch_batch"]
		
		if self.model_name == "TopologicalAE":
			return ["pytorch_batch", "LifeLong"]
		

	def create_dataset(self, input_data: InputData=None):
		
		if input_data is not None:
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]) if input_data.transform_data else torchvision.transforms.ToTensor()
		else:
			transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

		if self.dataset_name=="MNIST":
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
			MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(10000)])
			MNIST_train_subset.targets=MNIST_train.targets[0:10000]
			MNIST_val_subset= torch.utils.data.dataset.Subset(MNIST_val,[i for i in range(10000)])
			MNIST_val_subset.targets=MNIST_val.targets[0:10000]		

			return MNIST_train_subset, MNIST_val_subset
		
		if self.dataset_name=="CIFAR10":
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
			CIFAR_train_subset= torch.utils.data.dataset.Subset(CIFAR_train,[i for i in range(10000)])
			CIFAR_train_subset.targets=torch.Tensor(CIFAR_train.targets[0:10000])
			CIFAR_val_subset= torch.utils.data.dataset.Subset(CIFAR_val,[i for i in range(10000)])
			CIFAR_val_subset.targets=torch.Tensor(CIFAR_val.targets[0:10000])	

			return CIFAR_train_subset, CIFAR_val_subset



	def run(self):
		"""
		Main function to run the training and plotting of the SOM/STM.
		"""
		torch.manual_seed(self.config.SEED)
		random.seed(self.config.SEED)
		np.random.seed(self.config.SEED)

		self.select_training(self.model)

