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
	M, N = 30, 20
	INPUT_DIM = (28,28)
	SEED = 13
	DECAY = 90 # good practice: decay about 90% of number of weights update
	SIGMA = 10
	BATCH_SIZE = 15
	EPOCHS_ONLINE = 1
	EPOCHS_SIMPLE_BATCH = 20
	EPOCHS_PYTORCH_BATCH = 1
	LLL_EPOCHS_PER_SUBSET = 1
	LLL_SUBSET_SIZE = 1
	LLL_DISJOINT = True
	LEARNING_RATE = 0.1

config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}


class MnistRunner(BaseRunner):

	def __init__(self, config: object, dataset_name: str, input_data: InputData):
		super().__init__(config=config, dataset_name=dataset_name, input_data=input_data)
		
	def create_dataset(self):
		"""
		"""
		# data in .data and labels in .targets
		MNIST_train = torchvision.datasets.MNIST(
			root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
			train=True,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		MNIST_val = torchvision.datasets.MNIST(
			root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
			train=False,
			download=True,
			transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), self.input_data.transform_data]),
		)
		target_points={i: torch.Tensor([random.randint(0, self.config.M-1), random.randint(0, self.config.N-1)]) for i in range(10)}
		MNIST_train_subset= torch.utils.data.dataset.Subset(MNIST_train,[i for i in range(1000)])
		MNIST_train_subset.targets=MNIST_train.targets[0:1000]
		return MNIST_train_subset, MNIST_val, target_points



config=Config
random.seed(config.SEED)
input_data=InputData((28,28),1,"Unit")
mnist_runner=MnistRunner(config=config, dataset_name="MNIST", input_data=input_data)
mnist_runner.run()


