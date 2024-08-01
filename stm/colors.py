import numpy as np
import torch
from stm import STM
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
    dataset = TensorDataset(torch.Tensor(colors), torch.Tensor([i for i in range(15)]))
    target_points={i: torch.Tensor([1.,1.]) for i in range(len(color_names))}
    target_points[1]=torch.Tensor([10.,15.])
    return dataset, target_points



def train_som(stm: STM, dataset: TensorDataset, config, wandb_log: bool):
    """
    Train the SOM based on the specified mode.
    
    Args:
        som (SOM): The Self-Organizing Map to train.
        dataset (TensorDataset): Dataset for training.
        train_mode (str): Training mode, either 'simple_batch', 'pytorch_batch', or 'online'.
        config (Config): Configuration object with training parameters.
    """
    if wandb_log:
        wandb.init(project='STM-colors', config= config_dict)

    stm.train_batch_pytorch(dataset, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS_PYTORCH_BATCH, learning_rate = config.LEARNING_RATE, wandb_log = wandb_log)




def main():
    """Main function to execute the training and plotting of the SOM."""
    args = parse_arguments()
    # Set random seed
    torch.manual_seed(Config.SEED)
    dataset, target_points = create_dataset()
    input_data=InputData(Config.INPUT_DIM) 
    stm = STM(Config.M, Config.N, input_data, decay_rate=Config.DECAY, target_points=target_points)
    train_som(stm, dataset, Config, args.wandb_log)

if __name__ == "__main__":
    main()
