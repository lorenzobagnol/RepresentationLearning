from som import SOM
import torchvision
from torch.utils.data import TensorDataset
import torch
import matplotlib.pyplot as plt
from inputdata import InputData
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
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", dest='wandb_log', help="A bool. If true log in wandb.", type=bool, default=False)
	return parser.parse_args()

def create_dataset(input_data: InputData):
	"""
	"""
	# data in .data and labels in .targets
	train_MNIST_dataset = torchvision.datasets.MNIST(
		root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
		train=True,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	MNIST_dataset = torchvision.datasets.MNIST(
		root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
	)
	subset_train = torch.utils.data.Subset(train_MNIST_dataset, indices=range(10000))

	return subset_train, MNIST_dataset

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

	som.train_batch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_SIMPLE_BATCH, wandb_log = wandb_log)
	
def main():
    """Main function to execute the training and plotting of the SOM."""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(Config.SEED)
    
    input_data=InputData(Config.INPUT_DIM) 
    train_data, val_data = create_dataset(input_data)
    som = SOM(Config.M, Config.N, input_data, decay_rate=Config.DECAY)
    train_som(som, train_data, Config, args.wandb_log)
    image_grid = som.create_image_grid()
    #Plot
    plt.imshow(image_grid)
    plt.title('MNIST SOM')
    plt.show()

if __name__ == "__main__":
    main()


