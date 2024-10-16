import numpy as np
import torch
from utils.inputdata import InputData
from torch.utils.data import TensorDataset
import random
from utils.runner import BaseRunner


class Config():
	"""Configuration class for setting constants."""
	M, N = 100, 100
	INPUT_DIM = 3
	SEED = 13
	DECAY = 1000
	SIGMA = 5
	BATCH_SIZE = 20
	EPOCHS_ONLINE = 100
	EPOCHS_SIMPLE_BATCH = 200
	EPOCHS_PYTORCH_BATCH = 400
	LLL_EPOCHS_PER_SUBSET = 40
	LLL_SUBSET_SIZE = 1
	LLL_DISJOINT = True
	LEARNING_RATE = 0.01

config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}

class colorsRunner(BaseRunner):

	def __init__(self, config: object, dataset_name: str, input_data: InputData):
		super().__init__(config=config, dataset_name=dataset_name, input_data=input_data)
		
	def create_dataset(self):
		"""
		Create a dataset of RGB colors.
		
		Returns:
			tuple: A tuple containing the TensorDataset and a list of color names.
		"""
		colors =  torch.Tensor([
			[0,0,0],
			[1,0,0],
			[0,1,0],
			[0,0,1],
			[1,1,0],
			[1,0,1],
			[0,1,1],
			[1,1,1]
		])
		color_names =[
			"Black",   
			"Red",     
			"Green",   
			"Blue",    
			"Yellow",  
			"Magenta", 
			"Cyan",    
			"White"    
		]
		samples=list()
		targets=list()
		for i, color in enumerate(colors):
			for _ in range(1000):
				# Create a sample by substituting 1 with N(0.9, 1.0) and 0 with N(0.0, 0.1)
				sample = torch.where(color == 1, torch.rand(color.size()) * 0.1 + 0.9, torch.rand(color.size()) * 0.1)
				samples.append(sample)
				targets.append(i)
				
		train_dataset = TensorDataset(torch.stack(samples), torch.Tensor(targets))
		train_dataset.targets = torch.Tensor(targets)
		val_dataset=train_dataset

		target_points=self.generate_equally_distributed_points(len(color_names))

		return train_dataset, val_dataset, target_points


config=Config
input_data=InputData((1,1),3,"RGB")
color_runner=colorsRunner(config=config, dataset_name="colors", input_data=input_data)
color_runner.run()