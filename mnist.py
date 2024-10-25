import torchvision
import torch
import random
import os
import numpy as np

from utils.inputdata import InputData
from utils.runner import BaseRunner
from utils.config import Config, SOMConfig, SimpleBatchConfig, PytorchBatchConfig, LifeLongConfig, OnlineConfig




class MnistRunner(BaseRunner):

	def __init__(self, config: Config, dataset_name: str, input_data: InputData):
		super().__init__(config=config, dataset_name=dataset_name, input_data=input_data)
		
	def create_dataset(self):
		"""
		"""
		# data in .data and labels in .targets
		MNIST_train = torchvision.datasets.MNIST(
			root=os.path.curdir,
			train=True,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		MNIST_val = torchvision.datasets.MNIST(
			root=os.path.curdir,
			train=False,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(10000)])
		MNIST_train_subset.targets=MNIST_train.targets[0:10000]
		MNIST_val_subset= torch.utils.data.dataset.Subset(MNIST_val,[i for i in range(10000)])
		MNIST_val_subset.targets=MNIST_val.targets[0:10000]		
	
		# target_points=self.generate_equally_distributed_points(10)
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
		points=np.int32(points*min(self.config.som_config.M, self.config.som_config.N))
		points.tolist()
		random.shuffle(points)
		target_points={k : torch.Tensor(v) for k,v in enumerate(points)}

		return MNIST_train_subset, MNIST_val_subset, target_points


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
# mnist_runner=MnistRunner(config=config, dataset_name="MNIST", input_data=input_data)
# mnist_runner.run()







import random
from concurrent.futures import ProcessPoolExecutor
from itertools import product

def run_experiment(alpha, beta):
    """
    Function to run the experiment with a given configuration.
    Args:
        alpha, beta (float): Alpha and Beta values for LifeLongConfig.
    
    Returns:
        str: Message indicating the experiment completed.
    """
    input_data = InputData((28, 28), 1, "Unit")

    # Creating a specific config with varying parameters for alpha and beta
    config = Config(
        SEED=13,
        som_config=SOMConfig(M=20, N=20, SIGMA=10),
    	lifelong_config=LifeLongConfig(ALPHA=alpha, BETA=beta, BATCH_SIZE=20, EPOCHS_PER_SUBSET=20, SUBSET_SIZE=1, DISJOINT_TRAINING=True, LR_GLOBAL_BASELINE=0.1, SIGMA_BASELINE=2, LEARNING_RATE=0.001),
        simple_batch_config=SimpleBatchConfig(EPOCHS=1, BATCH_SIZE=20, BETA=0.01),
        pytorch_batch_config=PytorchBatchConfig(EPOCHS=1, BATCH_SIZE=20, LEARNING_RATE=0.001, BETA=0.01),
        online_config=OnlineConfig(EPOCHS=1)
    )

    random.seed(config.SEED)

    # Running the experiment
    mnist_runner = MnistRunner(config=config, dataset_name="MNIST", input_data=input_data)
    mnist_runner.run()

    return f"Experiment alpha={alpha}, beta={beta} completed."



MNIST_train = torchvision.datasets.MNIST(
			root=os.path.curdir,
			train=True,
			download=True,
			transform=torchvision.transforms.ToTensor(),
		)
MNIST_val = torchvision.datasets.MNIST(
	root=os.path.curdir,
	train=False,
	download=True,
	transform=torchvision.transforms.ToTensor(),
)

alphas = [5, 2, 1]  # 3 different alpha values
betas = [2, 0.25, 0.1]  # 3 different beta values

# Create 9 combinations of alpha and beta values
param_combinations = list(product(alphas, betas))

# Run experiments in parallel using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=9) as executor:
    futures = [
        executor.submit(run_experiment, alpha, beta)
        for (alpha, beta) in param_combinations
    ]
    
    # Wait for all futures to complete and print results
    for future in futures:
        print(future.result())

