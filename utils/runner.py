import torch.utils
import torch.utils.data
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import matplotlib.pyplot as plt
from utils.inputdata import InputData
import argparse
import wandb
import random
import math
from abc import ABC, abstractmethod
from typing import Any, Union

from models.som import SOM
from models.stm import STM
from utils.trainer import SOMTrainer
from utils.config import Config

class BaseRunner(ABC):

	def __init__(self, config: Config, dataset_name: str, input_data: InputData):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name
		self.input_data=input_data


	def generate_equally_distributed_points(self, P: int) -> dict[int, Tensor]:
		m=self.config.som_config.M
		n=self.config.som_config.N
		# Adjust M and N to exclude the borders (from 1 to M-2 and 1 to N-2)
		if m <= 2 or n <= 2:
			raise ValueError("Grid is too small to exclude borders.")
		# Compute the best possible factors for k_x and k_y, excluding borders
		k_x = int(math.sqrt(P * (m - 2) / (n - 2)))  # Approximate number of points in the x (row) direction
		k_y = int(math.sqrt(P * (n - 2) / (m - 2)))  # Approximate number of points inself.config. the y (col) direction
		# Adjust k_x and k_y to ensure the total number of points is at least P
		while k_x * k_y < P:
			if k_x < k_y:
				k_x += 1
			else:
				k_y += 1
		# Compute step sizes in the inner grid (excluding borders)
		step_x = (m - 2 - 1) / (k_x - 1) if k_x > 1 else 0
		step_y = (n - 2 - 1) / (k_y - 1) if k_y > 1 else 0
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
		parser.add_argument("--training", dest='training_mode', help="The training mode. Could be 'simple_batch', 'online', 'pytorch_batch', 'LifeLong'", type=str, default=None)
		parser.add_argument("--log", dest='wandb_log', help="Add '--log' to log in wandb.", action='store_true')
		return parser.parse_args()
	
	@abstractmethod
	def create_dataset(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, dict[int, torch.Tensor]]:
		pass

	def select_training(self, model: Union[SOM, STM], dataset_train: TensorDataset, dataset_val: TensorDataset, wandb_log: bool, train_mode: str = None):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		
		trainer = SOMTrainer(model, wandb_log, True)
		if train_mode==None:
			while True:
				train_mode=input("Choose a training mode. Could be one of "+ str(trainer.available_training_modes()))
				if train_mode in trainer.available_training_modes():
					break
		if train_mode not in trainer.available_training_modes():
			print("wrong training mode selected")
		model_name="STM" if type(model) is STM else "SOM"
		if wandb_log:
			wandb.init(project=model_name+'-'+self.dataset_name, job_type= train_mode)
		print("You have choose to train a SOM model with "+train_mode+" mode.")
		training_function = getattr(trainer, "train_"+train_mode)
		match train_mode:
			case "simple_batch":
				training_function(dataset_train, dataset_val, **self.config.simple_batch_config.to_dict())
			case "online":
				training_function(dataset_train, dataset_val, **self.config.online_config.to_dict())
			case "pytorch_batch":
				training_function(dataset_train, dataset_val, **self.config.pytorch_batch_config.to_dict())
			case "LifeLong":
				training_function(dataset_train, dataset_val, **self.config.lifelong_config.to_dict())

	def run(self):
		"""
		Main function to run the training and plotting of the SOM/STM.
		"""
		torch.manual_seed(self.config.SEED)
		random.seed(self.config.SEED)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		args = self.parse_arguments()
		dataset_train, dataset_val, target_points = self.create_dataset()
		match args.model:
			case "som":
				model = SOM(self.config.som_config.M, self.config.som_config.N, self.input_data, self.config.som_config.SIGMA)#.to(device)
			case "stm":
				model = STM(self.config.som_config.M, self.config.som_config.N, self.input_data, target_points=target_points, sigma= self.config.som_config.SIGMA)#.to(device)
		self.select_training(model, dataset_train, dataset_val, args.wandb_log, args.training_mode)

