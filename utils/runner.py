import torch.utils
import torch.utils.data
from torch.utils.data import TensorDataset
from torch import Tensor
import torch
import numpy as np
from utils.inputdata import InputData
import argparse
import wandb
import random
import math
from typing import Any, Union

from models.som import SOM
from models.stm import STM
from utils.trainer import SOMTrainer
from utils.config import Config

class Runner():

	def __init__(self, config: Config, dataset_name: str, input_data: InputData, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, model: SOM = None, training_mode: str= None, wandb: bool= None):
		super().__init__()
		self.config=config
		self.dataset_name=dataset_name
		self.input_data=input_data
		self.dataset_train=train_dataset
		self.dataset_val=val_dataset
		if (wandb==None or training_mode==None or model==None):
			args = self.parse_arguments()
			self.wandb_log=args.wandb_log
			self.training_mode=args.training_mode
			self.model_name=args.model
		else:
			self.wandb_log=wandb
			self.training_mode=training_mode
			self.model_name=model


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
	
	def generate_equally_distributed_points_v2(self, P: int=None) -> dict[int, Tensor]:
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
		points_list=np.int32(points*min(self.config.som_config.M, self.config.som_config.N)).tolist()
		random.seed(13)
		random.shuffle(points_list)
		dict_points={k : torch.Tensor(v) for k,v in enumerate(points_list)}
		return dict_points

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
		parser.add_argument("--model", dest='model', help="The model to run. Could be 'som', 'stm' or 'AE'", type=str, required=True)
		parser.add_argument("--training", dest='training_mode', help="The training mode. Could be 'simple_batch', 'online', 'pytorch_batch', 'LifeLong'", type=str, default=None)
		parser.add_argument("--log", dest='wandb_log', help="Add '--log' to log in wandb.", action='store_true')
		return parser.parse_args()


	def select_training(self, model: Union[SOM, STM]):
		"""
		Train the SOM based on the specified mode.
		
		Args:
			som (SOM): The Self-Organizing Map to train.
			dataset (TensorDataset): Dataset for training.
			train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		"""
		
		trainer = SOMTrainer(model, self.wandb_log, True)
		if self.training_mode==None:
			while True:
				train_mode=input("Choose a training mode. Could be one of "+ str(trainer.available_training_modes()))
				if train_mode in trainer.available_training_modes():
					self.training_mode=train_mode
					break
		if self.training_mode not in trainer.available_training_modes():
			print("wrong training mode selected")
			return
		model_name="STM" if type(model) is STM else "SOM"
		if self.wandb_log:
			wandb.init(project=model_name+'-'+self.dataset_name, job_type= self.training_mode)
		print("You have choose to train a "+model_name+" model with "+self.training_mode+" mode.")
		training_function = getattr(trainer, "train_"+self.training_mode)
		match self.training_mode:
			case "simple_batch":
				training_function(self.dataset_train, self.dataset_val, **self.config.simple_batch_config.to_dict(), **self.config.som_config.to_dict())
			case "online":
				training_function(self.dataset_train, self.dataset_val, **self.config.online_config.to_dict(), **self.config.som_config.to_dict())
			case "pytorch_batch":
				training_function(self.dataset_train, self.dataset_val, **self.config.pytorch_batch_config.to_dict(), **self.config.som_config.to_dict())
			case "LifeLong":
				training_function(self.dataset_train, self.dataset_val, **self.config.lifelong_config.to_dict(), **self.config.som_config.to_dict())
		return

	def run(self):
		"""
		Main function to run the training and plotting of the SOM/STM.
		"""
		torch.manual_seed(self.config.SEED)
		random.seed(self.config.SEED)
		np.random.seed(self.config.SEED)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		match self.model_name:
			case "som":
				model = SOM(self.config.som_config.M, self.config.som_config.N, self.input_data, self.config.som_config.SIGMA).to(device)
			case "stm":
				target_points=self.generate_equally_distributed_points_v2()
				model = STM(self.config.som_config.M, self.config.som_config.N, self.input_data, target_points=target_points, sigma= self.config.som_config.SIGMA).to(device)
		self.select_training(model)

