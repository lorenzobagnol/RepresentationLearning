import random
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import torchvision
import torch
import os
import traceback
import torch.multiprocessing as mp

from utils.inputdata import InputData
from utils.runner import Runner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig
		
def create_dataset(input_data: InputData):
	"""
	"""
	# data in .data and labels in .targets
	MNIST_train = torchvision.datasets.MNIST(
		root=os.path.curdir,
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	MNIST_val = torchvision.datasets.MNIST(
		root=os.path.curdir,
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(10000)])
	MNIST_train_subset.targets=MNIST_train.targets[0:10000]
	MNIST_val_subset= torch.utils.data.dataset.Subset(MNIST_val,[i for i in range(10000)])
	MNIST_val_subset.targets=MNIST_val.targets[0:10000]		

	return MNIST_train_subset, MNIST_val_subset


# input_data=InputData((28,28),1,"Unit")
# config = Config(
# 	SEED=13,
#     som_config=SOMConfig(M=20, N=20, SIGMA=10),
#     lifelong_config=LifeLongConfig(ALPHA=10, BETA=0.005, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
#     simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
#     pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
#     online_config=OnlineConfig(EPOCHS=1)
# )
# random.seed(config.SEED)
# dataset_train, dataset_val = create_dataset(input_data=input_data)
# mnist_runner=Runner(config=config, dataset_name="MNIST", input_data=input_data, train_dataset=dataset_train, val_dataset=dataset_val)
# mnist_runner.run()








def run_experiment(alpha, beta, input_data, dataset_train, dataset_val):
	"""
	Function to run the experiment with a given configuration.
	Args:
		alpha, beta (float): Alpha and Beta values for LifeLongConfig.
	
	Returns:
		str: Message indicating the experiment completed.
	"""
	try:
		# Creating a specific config with varying parameters for alpha and beta
		config = Config(
			SEED=13,
			som_config=SOMConfig(M=20, N=20, SIGMA=10),
			lifelong_config=LifeLongConfig(ALPHA=alpha, BETA=beta, BATCH_SIZE=20, EPOCHS_PER_SUBSET=200, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001, MODE=""),
			simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
			pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
			online_config=OnlineConfig(EPOCHS=1)
		)

		random.seed(config.SEED)

		# Running the experiment
		mnist_runner=Runner(config=config, dataset_name="MNIST", input_data=input_data, train_dataset=dataset_train, val_dataset=dataset_val, model="stm", training_mode="LifeLong", wandb=True)
		mnist_runner.run()

		return f"Experiment alpha={alpha}, beta={beta} completed."
	except Exception as e:
		error_message = f"Error in experiment alpha={alpha}, beta={beta}: {e}\n{traceback.format_exc()}"
		print(error_message)
		return error_message


if __name__ == '__main__':

	mp.set_start_method('spawn', force=True)

	input_data = InputData((28, 28), 1, "Unit")
	dataset_train, dataset_val = create_dataset(input_data=input_data)

	alphas = [5, 2, 1]  # 3 different alpha values
	betas = [0.005, 0.01, 0.02]  # 3 different beta values

	# Create 9 combinations of alpha and beta values
	param_combinations = list(product(alphas, betas))

	# Run experiments in parallel using ProcessPoolExecutor
	with ProcessPoolExecutor(max_workers=1) as executor:
		futures = [
			executor.submit(run_experiment, alpha, beta, input_data, dataset_train, dataset_val)
			for (alpha, beta) in param_combinations
		]
		
		# Wait for all futures to complete and print results
		for future in futures:
			print(future.result())

