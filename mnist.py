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
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig, WandBConfig, VARS
		


if __name__ == '__main__':

	# Creating a specific config with varying parameters for alpha and var2
	config = Config(
		SEED=13,
		som_config=SOMConfig(M=20, N=20, INPUT_DATA=InputData((28,28), channels=1, channel_range="RGB")),
		LifeLong_config=LifeLongConfig(SIGMA=10, ALPHA=None, BETA=0.02, BATCH_SIZE=2, EPOCHS_PER_SUBSET=2, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=1.5, LEARNING_RATE=0.1, MODE="Base"),
		simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01, SIGMA=10),
		pytorch_batch_config=PytorchBatchConfig(EPOCHS=200, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01, SIGMA=10, MODE="Base"),
		online_config=OnlineConfig(EPOCHS=1, SIGMA=10),
		weights_and_biases_config=WandBConfig(PROJECT="prova"),
		variables=VARS(SEED=10, target_radius=1.5)
	)

	runner=Runner(config=config, dataset_name="MNIST")
	
	# vars0 = [5, 1, 0.2, 0]  # 3 different alpha values
	vars1 = list(range(1))  # 10 different seeds
	vars2 = ["Base"]  # different mode values

	# Create 9 combinations of alpha and beta values
	param_combinations = list(product(vars1, vars2))

	# # use this line if using ProcessPoolExecutor
	# mp.set_start_method('spawn', force=True)
	# # Run experiments in parallel using ProcessPoolExecutor
	# with ProcessPoolExecutor() as executor:
	# 	futures = [
	# 		executor.submit(run_experiment, var1, var2, var3, dataset_train=dataset_train, dataset_val=dataset_val)
	# 		for (var1, var2, var3) in param_combinations
	# 	]
		
	# 	# Wait for all futures to complete and print results
	# 	for future in futures:
	# 		print(future.result())

	for (var1, var2) in param_combinations:
		runner.run()

