import pickle
import os
import numpy as np
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torchvision
from torch.utils.data import TensorDataset
import torch
from utils.inputdata import InputData
import random
from utils.runner import BaseRunner

class Config():
	"""Configuration class for setting constants."""
	M, N = 10, 10
	INPUT_DIM = (32,32)
	SEED = 13
	DECAY = 90 # good practice: decay about 90% of number of weights update
	SIGMA = 3
	BATCH_SIZE = 20
	EPOCHS_ONLINE = 1
	EPOCHS_SIMPLE_BATCH = 20
	EPOCHS_PYTORCH_BATCH = 40
	LLL_EPOCHS_PER_SUBSET = 80
	LLL_SUBSET_SIZE = 1
	LLL_DISJOINT = True
	LEARNING_RATE = 0.01

config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}


class CifarRunner(BaseRunner):

	def __init__(self, config: object, dataset_name: str, input_data: InputData):
		super().__init__(config=config, dataset_name=dataset_name, input_data=input_data)
		
	def create_dataset(self):
		"""
		"""
		all_data = []
		all_labels = []
		data_path = "cifar-10-batches-py"


		for file in os.listdir(data_path):
			if file.startswith("data"):
				file_path = os.path.join(data_path, file) 
				with open(file_path, 'rb') as fo:
					batch_dict = pickle.load(fo, encoding='bytes')
					all_data.append(batch_dict[b'data'])
					all_labels.extend(batch_dict[b'labels'])
		combined_data = torch.Tensor(np.vstack(all_data))
		combined_labels = torch.Tensor(all_labels)

		# data in .data and labels in .targets
		transformed_data = self.input_data.transform_dataset(combined_data)
		cifar_train = torch.utils.data.TensorDataset(transformed_data,combined_labels)
		cifar_val = None
		cifar_train_subset= torch.utils.data.dataset.Subset(cifar_train,[i for i in range(10000)])
		cifar_train.targets = torch.Tensor(combined_labels)
		cifar_train_subset.targets=cifar_train.targets[0:10000]
		cifar_val_subset= None
	
		target_points=self.generate_equally_distributed_points(10)

		return cifar_train_subset, cifar_val_subset, target_points



config=Config
random.seed(config.SEED)
input_data=InputData(config.INPUT_DIM,3,"Unit")
cifar_runner=CifarRunner(config=config, dataset_name="cifar", input_data=input_data)
cifar_runner.run()


