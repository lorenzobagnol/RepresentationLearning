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

	def __init__(self, config, dataset_name: str, input_data: InputData):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name
		self.input_data=input_data


	def generate_equally_distributed_points(self, n_points):
		grid_div_x = int((n_points * self.config.M / self.config.N) ** 0.5)  # Divisions along the x-axis (height)
		grid_div_y = int(n_points / grid_div_x)  # Divisions along the y-axis (width)

		# Handle edge cases where the divisions might be too small or large
		if grid_div_x == 0: grid_div_x = 1
		if grid_div_y == 0: grid_div_y = 1 # Number of divisions along the y-axis (N dimension)
		if grid_div_x * grid_div_y < n_points:
			grid_div_y += 1
		grid_step_x = self.config.M / grid_div_x  # Step size in the x direction
		grid_step_y = self.config.N / grid_div_y  # Step size in the y direction
		
		points = []
		stop_loop=False
		for i in range(grid_div_x):
			for j in range(grid_div_y):
				# Generate one random point in each grid cell
				x = random.randint(i * int(grid_step_x), min(int((i + 1) * grid_step_x) - 1, self.config.M - 1))
				y = random.randint(j * int(grid_step_y), min(int((j + 1) * grid_step_y) - 1, self.config.N - 1))
				points.append([x, y])
				if len(points) >= n_points:
					stop_loop=True
					break
			if stop_loop:
				break
		dict_points={k : torch.Tensor(v) for k,v in enumerate(points)}
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
