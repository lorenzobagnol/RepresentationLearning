import random
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import torchvision
import torch
import os
import traceback
import torch.multiprocessing as mp
import wandb

from utils.inputdata import InputData
from utils.runner import Runner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig, VARS
		
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



def run_experiment(*args, input_data, dataset_train, dataset_val):
	"""
	Function to run the experiment with a given configuration.
	
	Returns:
		str: Message indicating the experiment completed.
	"""
	try:
		wandb_log = True
		if wandb_log:
			entity = "replearn"
			project = "STM-competence-decay-MNIST"
			api = wandb.Api()
			filters = {

				"state": "finished",
				"config.target_radius": args[0],
				"config.MODE": args[1]
			}
			runs = api.runs(
				path=f"{entity}/{project}",
				filters=filters,
				order="-created_at"
			)

			if len(runs)!=0:
				return f"run already exists"


		# Creating a specific config with varying parameters for alpha and var2
		config = Config(
			SEED=13,
			som_config=SOMConfig(M=20, N=20, SIGMA=10),
			lifelong_config=LifeLongConfig(ALPHA=None, BETA=0.02, BATCH_SIZE=20, EPOCHS_PER_SUBSET=200, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=1.5, LEARNING_RATE=0.1, MODE=args[1]),
			simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
			pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
			online_config=OnlineConfig(EPOCHS=1),
			variables=VARS(target_radius=args[0])
		)
		random.seed(config.SEED)

		# Running the experiment
		mnist_runner=Runner(config=config, dataset_name="competence-decay-MNIST", input_data=input_data, train_dataset=dataset_train, val_dataset=dataset_val, model="stm", training_mode="LifeLong", wandb=wandb_log)
		mnist_runner.run()

		return f"Experiment "+str(list(args))+" completed."
	except Exception as e:
		error_message = f"Error in experiment "+str(list(args))+f": {e}\n{traceback.format_exc()}"
		print(error_message)
		return error_message


if __name__ == '__main__':

	# os.environ["CUDA_VISIBLE_DEVICES"]="0"
	input_data = InputData((28, 28), 1, "Unit")
	dataset_train, dataset_val = create_dataset(input_data=input_data)

	# vars0 = [5, 1, 0.2, 0]  # 3 different alpha values
	vars1 = [10, 5, 2, 1.5]  # 3 different targed radius values
	vars2 = ["Base", "STC-modified", "Base_Norm"]  # different mode values

	# Create 9 combinations of alpha and beta values
	param_combinations = list(product(vars1, vars2))

	# # use this line if using ProcessPoolExecutor
	# mp.set_start_method('spawn', force=True)
	# # Run experiments in parallel using ProcessPoolExecutor
	# with ProcessPoolExecutor() as executor:
	# 	futures = [
	# 		executor.submit(run_experiment, var1, var2, var3, input_data=input_data, dataset_train=dataset_train, dataset_val=dataset_val)
	# 		for (var1, var2, var3) in param_combinations
	# 	]
		
	# 	# Wait for all futures to complete and print results
	# 	for future in futures:
	# 		print(future.result())

	for (var2, var3) in param_combinations:
		run_experiment( var2, var3, input_data=input_data, dataset_train=dataset_train, dataset_val=dataset_val)

