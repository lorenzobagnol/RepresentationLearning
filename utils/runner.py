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
import math
from abc import ABC, abstractmethod
from typing import Any


class BaseRunner(ABC):

	def __init__(self, config, dataset_name: str, input_data: InputData):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name
		self.input_data=input_data


	def generate_equally_distributed_points(self, P: int):
		# Adjust M and N to exclude the borders (from 1 to M-2 and 1 to N-2)
		if self.config.M <= 2 or self.config.N <= 2:
			raise ValueError("Grid is too small to exclude borders.")

		# Compute the best possible factors for k_x and k_y, excluding borders
		k_x = int(math.sqrt(P * (self.config.M - 2) / (self.config.N - 2)))  # Approximate number of points in the x (row) direction
		k_y = int(math.sqrt(P * (self.config.N - 2) / (self.config.M - 2)))  # Approximate number of points inself.config. the y (col) direction

		# Adjust k_x and k_y to ensure the total number of points is at least P
		while k_x * k_y < P:
			if k_x < k_y:
				k_x += 1
			else:
				k_y += 1

		# Compute step sizes in the inner grid (excluding borders)
		step_x = (self.config.M - 2 - 1) / (k_x - 1) if k_x > 1 else 0
		step_y = (self.config.N - 2 - 1) / (k_y - 1) if k_y > 1 else 0

		# Generate points in the range [1, M-2] and [1, N-2] to avoid borders
		x_coords = [round(1 + i * step_x) for i in range(k_x)]
		y_coords = [round(1 + j * step_y) for j in range(k_y)]

		# Combine x and y coordinates to get the points
		points = [(x, y) for x in x_coords for y in y_coords]
		dict_points={k : torch.Tensor(v) for k,v in enumerate(points[:P])}
		return dict_points


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
	def create_dataset(self) -> tuple[Any, torch.utils.data.Dataset, dict[int, torch.Tensor]]:
		pass

	def train_som(self, som: SOM, dataset: TensorDataset, wandb_log: bool):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		config_dict={key: value for key, value in self.config.__dict__.items() if not key.startswith('_')}
		
		while True:
			train_mode=input("Choose a training mode. Could be: 'simple_batch', 'pytorch_batch' or 'online' ")
			if train_mode == "simple_batch":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= config_dict, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_batch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_SIMPLE_BATCH, decay_rate=self.config.DECAY, wandb_log = wandb_log, clip_images=True)
				break
			elif train_mode == "pytorch_batch":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= config_dict, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_batch_pytorch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_PYTORCH_BATCH, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log, clip_images=True)
				break
			elif train_mode == "online":
				if wandb_log:
					wandb.init(project='SOM-'+self.dataset_name, config= config_dict, job_type= train_mode)
				print("You have choose to train a SOM model with "+train_mode+" mode.")
				som.train_online(dataset, epochs = self.config.EPOCHS_ONLINE, decay_rate=self.config.DECAY, alpha=self.config.LEARNING_RATE, wandb_log = wandb_log, clip_images=True)
				break

	def train_stm(self, stm: STM, dataset: TensorDataset, wandb_log: bool):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		config_dict={key: value for key, value in self.config.__dict__.items() if not key.startswith('_')}

		while True:
			train_mode=input("Choose a training mode. Could be: 'pytorch_batch' or 'LifeLong_learning' ")
			if train_mode == "pytorch_batch":
				if wandb_log:
					wandb.init(project='STM-'+self.dataset_name, config= config_dict, job_type= train_mode)
				print("\nYou have choose to train a STM model with "+train_mode+" mode.")
				stm.train_batch_pytorch(dataset, batch_size = self.config.BATCH_SIZE, epochs = self.config.EPOCHS_PYTORCH_BATCH, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log, clip_images=True)
				break
			if train_mode == "LifeLong_learning":
				if wandb_log:
					wandb.init(project='STM-'+self.dataset_name, config= config_dict, job_type= train_mode)
				print("\nYou have choose to train a STM model with "+train_mode+" mode.")
				stm.train_lifelong(dataset, batch_size = self.config.BATCH_SIZE, subset_size = self.config.LLL_SUBSET_SIZE, epochs_per_subset =  self.config.LLL_EPOCHS_PER_SUBSET, disjoint_training =  self.config.LLL_DISJOINT, learning_rate = self.config.LEARNING_RATE, decay_rate=self.config.DECAY, wandb_log = wandb_log, clip_images=True)
				break

	def run(self):
		"""
		Main function to run the training and plotting of the SOM/STM.
		"""
		torch.manual_seed(self.config.SEED)
		random.seed(self.config.SEED)

		args = self.parse_arguments()
		dataset_train, dataset_val, target_points = self.create_dataset()
		match args.model:
			case "som":
				som = SOM(self.config.M, self.config.N, self.input_data, self.config.SIGMA)
				self.train_som(som, dataset_train, args.wandb_log)
				image_grid = som.create_image_grid()
				fig=som.resize_image(image_grid)
			case "stm":
				stm = STM(self.config.M, self.config.N, self.input_data, target_points=target_points, sigma= self.config.SIGMA)
				self.train_stm(stm, dataset_train, args.wandb_log)
				image_grid = stm.create_image_grid()
				fig=stm.resize_image_add_target_points(image_grid)
		plt.show()
