import torch
import random

from utils.inputdata import InputData
from torch.utils.data import TensorDataset
from utils.runner import BaseRunner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig


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


input_data=InputData((1,1),3,"RGB")
config = Config(
	SEED=13,
    som_config=SOMConfig(M=20, N=20, SIGMA=10),
    lifelong_config=LifeLongConfig(ALPHA=10, BETA=0.01, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
    simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
    pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
    online_config=OnlineConfig(EPOCHS=1)
)
random.seed(config.SEED)
color_runner=colorsRunner(config=config, dataset_name="colors", input_data=input_data)
color_runner.run()