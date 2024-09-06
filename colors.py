import numpy as np
import torch
from utils.inputdata import InputData
from torch.utils.data import TensorDataset
import random
from utils.runner import BaseRunner


class Config:
	"""Configuration class for setting constants."""
	M, N = 50, 50
	INPUT_DIM = 3
	SEED = 13
	DECAY = 10
	SIGMA = 7
	BATCH_SIZE = 15
	EPOCHS_ONLINE = 100
	EPOCHS_SIMPLE_BATCH = 200
	EPOCHS_PYTORCH_BATCH = 400
	LLL_EPOCHS_PER_SUBSET = 40
	LLL_SUBSET_SIZE = 1
	LLL_DISJOINT = True
	LEARNING_RATE = 0.1
config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}

class colorsRunner(BaseRunner):

	def __init__(self, config: object, dataset_name: str):
		super().__init__(config=config, dataset_name=dataset_name)
		
	def create_dataset(self, input_data: InputData = None):
		"""
		Create a dataset of RGB colors.
		
		Returns:
			tuple: A tuple containing the TensorDataset and a list of color names.
		"""
		colors = np.array([
			[0., 0., 0.],
			[0., 0., 1.],
			[0., 0., 0.5],
			[0.125, 0.529, 1.0],
			[0.33, 0.4, 0.67],
			[0.6, 0.5, 1.0],
			[0., 1., 0.],
			[1., 0., 0.],
			[0., 1., 1.],
			[1., 0., 1.],
			[1., 1., 0.],
			[1., 1., 1.],
			[.33, .33, .33],
			[.5, .5, .5],
			[.66, .66, .66]
		])
		color_names = [
			'black', 'blue', 'darkblue', 'skyblue',
			'greyblue', 'lilac', 'green', 'red',
			'cyan', 'violet', 'yellow', 'white',
			'darkgrey', 'mediumgrey', 'lightgrey'
		]
		train_dataset = TensorDataset(torch.Tensor(colors), torch.Tensor([i for i in range(15)]))
		train_dataset.targets = torch.IntTensor([i for i in range(15)])
		target_points={i: torch.Tensor([random.randint(0, Config.M), random.randint(0, Config.N)]) for i in range(len(color_names))}
		val_dataset=train_dataset
		return train_dataset, val_dataset, target_points



color_runner=colorsRunner(config=Config, dataset_name="colors")
color_runner.run()