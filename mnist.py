import torchvision
from torch.utils.data import TensorDataset
import torch
from utils.inputdata import InputData
import random
from utils.runner import BaseRunner

class Config:
	"""Configuration class for setting constants."""
	M, N = 10, 10
	INPUT_DIM = (28,28)
	SEED = 13
	DECAY = 360 # good practice: decay about 90% of number of weights update
	SIGMA = 7
	BATCH_SIZE = 15
	EPOCHS_ONLINE = 100
	EPOCHS_SIMPLE_BATCH = 200
	EPOCHS_PYTORCH_BATCH = 100
	LLL_EPOCHS_PER_SUBSET = 40
	LLL_SUBSET_SIZE = 1
	LLL_DISJOINT = True
	LEARNING_RATE = 0.1

config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}


class MnistRunner(BaseRunner):

	def __init__(self, config: object, dataset_name: str):
		super().__init__(config=config, dataset_name=dataset_name)
		
	def create_dataset(self, input_data: InputData):
		"""
		"""
		# data in .data and labels in .targets
		MNIST_train = torchvision.datasets.MNIST(
			root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
			train=True,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
		)
		MNIST_val = torchvision.datasets.MNIST(
			root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
			train=False,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
		)
		MNIST_train = torch.utils.data.Subset(MNIST_train, indices=range(10000))
		target_points={i: torch.Tensor([random.randint(0, Config.M), random.randint(0, Config.N)]) for i in range(10)}
		target_points[1]=torch.Tensor([9., 1.])

		return MNIST_train, MNIST_val, target_points




mnist_runner=MnistRunner(config=Config, dataset_name="MNIST")
mnist_runner.run()


