import numpy as np
import torch
from som import SOM
import matplotlib.pyplot as plt
from inputdata import InputData
from torch.utils.data import TensorDataset
import argparse
import wandb

class Config:
    """Configuration class for setting constants."""
    M, N = 20, 30
    INPUT_DIM = 3
    SEED = 13
    DECAY = 360 # good practice: decay about 90% of number of weights update
    BATCH_SIZE = 15
    EPOCHS_SIMPLE_BATCH = 200
    EPOCHS_PYTORCH_BATCH = 400
    LEARNING_RATE = 0.01
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
    parser.add_argument("--mode", dest='train_mode', help="Training mode. Could be: 'simple_batch', 'pytorch_batch' or 'online'",
                        choices=['simple_batch', 'pytorch_batch', 'online'], type=str, required=True)
    return parser.parse_args()

def create_dataset():
    """
    Create a dataset of RGB colors.
    
    Returns:
        tuple: A tuple containing the TensorDataset and a list of color names.
    """
    colors = np.array([
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.5],
        [0.125, 0.529, 1.0],
        [0.33, 0.4, 0.67],
        [0.6, 0.5, 1.0],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [.33, .33, .33],
        [.5, .5, .5],
        [.66, .66, .66]
    ])
    color_names = [
        'black', 'blue', 'darkblue', 'skyblue',
        'greyblue', 'lilac', 'green', 'red',
        'cyan', 'violet', 'yellow', 'white',
        'darkgrey', 'mediumgrey', 'lightgrey'
    ]
    dataset = TensorDataset(torch.Tensor(colors))
    return dataset, color_names



def train_som(som: SOM, dataset: TensorDataset, train_mode: str, config, wandb_log: bool):
    """
    Train the SOM based on the specified mode.
    
    Args:
        som (SOM): The Self-Organizing Map to train.
        dataset (TensorDataset): Dataset for training.
        train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
        config (Config): Configuration object with training parameters.
    """
    if wandb_log:
        wandb.init(project='SOM-colors', config= config_dict)

    if train_mode == "simple_batch":
        som.train_batch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_SIMPLE_BATCH, wandb_log = wandb_log)
    elif train_mode == "pytorch_batch":
        som.train_batch_pytorch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_PYTORCH_BATCH, learning_rate = config.LEARNING_RATE, wandb_log = wandb_log)
    elif train_mode == "online":
        som.train_online(dataset, epochs = config.EPOCHS_ONLINE, wandb_log = wandb_log)





def main():
    """Main function to execute the training and plotting of the SOM."""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(Config.SEED)
    
    dataset, color_names = create_dataset()
    input_data=InputData(Config.INPUT_DIM) 
    som = SOM(Config.M, Config.N, input_data, decay_rate=Config.DECAY)
    train_som(som, dataset, args.train_mode, Config, args.wandb_log)
    image_grid = som.create_image_grid()
    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    plt.show()

if __name__ == "__main__":
    main()