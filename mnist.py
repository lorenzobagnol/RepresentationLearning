from models.som import SOM
from models.stm import STM
import torchvision
from torch.utils.data import TensorDataset
import torch
import matplotlib.pyplot as plt
from utils.inputdata import InputData
import argparse
import wandb

class Config:
	"""Configuration class for setting constants."""
	M, N = 10, 10
	INPUT_DIM = (28,28)
	SEED = 13
	DECAY = 360 # good practice: decay about 90% of number of weights update
	BATCH_SIZE = 15
	EPOCHS_SIMPLE_BATCH = 200
	EPOCHS_PYTORCH_BATCH = 100
	LEARNING_RATE = 0.1
	EPOCHS_ONLINE = 100
config_dict={key: value for key, value in Config.__dict__.items() if not key.startswith('_')}

def parse_arguments():
	"""
	Parse command line arguments.
	
	Returns:
		argparse.Namespace: Parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(
                    prog='Color SOM/STM training',
                    description='this script can train a SOM or a STM with MNIST dataset',
                    epilog='Text at the bottom of help')
	parser.add_argument("--model", dest='model', help="The model to run. Could be 'som', 'stm' or 'AE'", type=str, required=True)
	parser.add_argument("--log", dest='wandb_log', help="A bool. If true log in wandb. Default set to False.", type=bool, default=False)
	
	return parser.parse_args()

def create_dataset(input_data: InputData):
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
	target_points={i: torch.Tensor([i, i]) for i in range(10)}
	target_points[1]=torch.Tensor([9., 1.])

	return MNIST_train, MNIST_val, target_points

def train_som(som: SOM, dataset: TensorDataset, config, wandb_log: bool):
	"""
	Train the SOM based on the specified mode.
	
	Args:
		som (SOM): The Self-Organizing Map to train.
		dataset (TensorDataset): Dataset for training.
		train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		config (Config): Configuration object with training parameters.
	"""
	if wandb_log:
		wandb.init(project='SOM-MNIST', config= config_dict)

	som.train_batch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_SIMPLE_BATCH, decay_rate=Config.DECAY, wandb_log = wandb_log)

def train_stm(stm: STM, dataset: TensorDataset, config, wandb_log: bool):
	"""
	Train the SOM based on the specified mode.
	
	Args:
		som (SOM): The Self-Organizing Map to train.
		dataset (TensorDataset): Dataset for training.
		train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
		config (Config): Configuration object with training parameters.
	"""
	if wandb_log:
		wandb.init(project='STM-MNIST', config= config_dict)
	
	stm.train_batch_pytorch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_PYTORCH_BATCH, learning_rate = config.LEARNING_RATE, decay_rate=Config.DECAY, wandb_log = wandb_log)


def main():
	"""Main function to execute the training and plotting of the SOM."""
	args = parse_arguments()
	
	# Set random seed
	torch.manual_seed(Config.SEED)
	
	input_data=InputData(Config.INPUT_DIM) 
	dataset_train, dataset_val, target_points = create_dataset(input_data)
	match args.model:
		case "som":
			som = SOM(Config.M, Config.N, input_data)
			train_som(som, dataset_train, Config, args.wandb_log)
			image_grid = som.create_image_grid()
		case "stm":
			stm = STM(Config.M, Config.N, input_data, target_points=target_points)
			train_stm(stm, dataset_train, Config, args.wandb_log)
			image_grid = stm.create_image_grid()
	#Plot
	plt.imshow(image_grid)
	plt.title('MNIST with '+ args.model)
	plt.show()

if __name__ == "__main__":
	main()


