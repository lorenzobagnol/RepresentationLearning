import torch.utils
import torch.utils.data
from models.som import SOM
from models.stm import STM
from torch.utils.data import TensorDataset
import torch
import matplotlib.pyplot as plt
from utils.inputdata import InputData
import argparse
import wandb
import random
from abc import ABC, abstractmethod
from typing import Any


class BaseRunner(ABC):

	def __init__(self, config: object, dataset_name: str):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name

	def parse_arguments(self):
		"""
		Parse command line arguments.
		
		Returns:
			argparse.Namespace: Parsed command line arguments.
		"""
		parser = argparse.ArgumentParser(
						prog='Color SOM/STM training',
						description='this script can train a SOM or a STM with MNIST dataset',
						epilog='Text at the bottom of help')
		parser.add_argument("--model", dest='model', help="The model to run. Could be 'som', 'stm' or 'AE'", type=str, required=True)
		parser.add_argument("--log", dest='wandb_log', help="Add '--log' to log in wandb.", action='store_true')
		return parser.parse_args()
	
	@abstractmethod
	def create_dataset(self, input_data: InputData = None) -> tuple[Any, torch.utils.data.Dataset, dict[int, torch.Tensor]]:
		pass

	def train_som(self, som: SOM, dataset: TensorDataset, wandb_log: bool):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
			config (Config): Configuration object with training parameters.
		"""
		while True:
			train_mode=input("Choose a training mode. Could be: 'simple_batch', 'pytorch_batch' or 'online' ")
			if train_mode == "simple_batch":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= self.config, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_batch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_SIMPLE_BATCH, decay_rate=self.config.DECAY, wandb_log = wandb_log)
				break
			elif train_mode == "pytorch_batch":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= self.config, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_batch_pytorch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_PYTORCH_BATCH, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log)
				break
			elif train_mode == "online":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= self.config, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_online(dataset, epochs = self.config.EPOCHS_ONLINE, decay_rate=self.config.DECAY, alpha=self.config.LEARNING_RATE, wandb_log = wandb_log)
				break

	def train_stm(self, stm: STM, dataset: TensorDataset, wandb_log: bool):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
			config (Config): Configuration object with training parameters.
		"""
		while True:
			train_mode=input("Choose a training mode. Could be: 'pytorch_batch' or 'LifeLong_learning' ")
			if train_mode == "pytorch_batch":
				if wandb_log:
					wandb.init(project='STM-'+self.dataset_name, config= self.config, job_type= train_mode)
				print("\nYou have choose to train a STM model with "+train_mode+" mode.")
				stm.train_batch_pytorch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_PYTORCH_BATCH, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log)
				break
			if train_mode == "LifeLong_learning":
				if wandb_log:
					wandb.init(project='STM-'+self.dataset_name, config= self.config, job_type= train_mode)
				print("\nYou have choose to train a STM model with "+train_mode+" mode.")
				stm.train_batch_pytorch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_PYTORCH_BATCH, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log)
				break

	def run(self):
		"""Main function to run the training and plotting of the SOM/STM."""
		args = self.parse_arguments()
		
		# Set random seed
		torch.manual_seed(self.config.SEED)
		random.seed=self.config.SEED
		input_data=InputData(self.config.INPUT_DIM) 
		dataset_train, dataset_val, target_points = self.create_dataset(input_data)
		match args.model:
			case "som":
				som = SOM(self.config.M, self.config.N, input_data)
				self.train_som(som, dataset_train, args.wandb_log)
				image_grid = som.create_image_grid()
			case "stm":
				stm = STM(self.config.M, self.config.N, input_data, target_points=target_points)
				self.train_stm(stm, dataset_train, args.wandb_log)
				image_grid = stm.create_image_grid()
		#Plot
		plt.imshow(image_grid)
		plt.title(self.dataset_name+' with '+ args.model)
		plt.show()
