import pickle
import os
import numpy as np
import torch
import random

from utils.inputdata import InputData
from utils.runner import BaseRunner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig





class CifarRunner(BaseRunner):

	def __init__(self, config: Config, dataset_name: str, input_data: InputData):
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



input_data=InputData((32,32),3,"Unit")
config = Config(
	SEED=13,
    som_config=SOMConfig(M=20, N=20, SIGMA=10),
    lifelong_config=LifeLongConfig(ALPHA=10, BETA=0.01, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
    simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20),
    pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001),
    online_config=OnlineConfig(EPOCHS=1)
)
random.seed(config.SEED)
cifar_runner=CifarRunner(config=config, dataset_name="cifar", input_data=input_data)
cifar_runner.run()


