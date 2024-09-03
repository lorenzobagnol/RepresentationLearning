from typing import Sequence, Union, Tuple, Generator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from utils.inputdata import InputData
from models.som_base import BaseSOM
from tqdm import tqdm
import wandb
import PIL

class SOM(BaseSOM):
	"""
	Self-Organizing Map with Gaussian Neighbourhood function.
	"""
	def __init__(self, m: int , n: int, input_data: InputData, alpha=None, sigma=None):
		"""
        Initialize the SOM.

        Args:
            m (int): Number of rows in the SOM grid.
            n (int): Number of columns in the SOM grid.
            input_data (InputData): InputData object containing input dimensions.
            alpha (float, optional): Initial learning rate. Defaults to 0.3.
            sigma (float, optional): Initial radius of the neighbourhood function. Defaults to half the maximum of m or n.
        """
		super().__init__(m = m, n = n, input_data = input_data, alpha = alpha, sigma = sigma)
	
	def train_online(self, training_set: datasets, epochs: int, decay_rate: int, wandb_log: bool = False):
		"""
        Train the SOM using online learning.

        Args:
            training_set (Dataset): Dataset for training.
            epochs (int): Number of epochs to train for.
			decay_rate (int): Decay rate for the learning rate.

        """
		print("\nSOM online-training is starting with hyper-parameters:")
		print("epochs = "+str(epochs)+"\n\n")
		for it in range(epochs):
			for i, el in tqdm(enumerate(training_set), f"epoch {it+1}", len(training_set)):
				x=el[0]
				# look for the best matching unit (BMU)
				dists = torch.pairwise_distance(x, self.weights, p=2)
				_, bmu_index = torch.min(dists, 0)
				bmu_loc = self.locations[bmu_index,:]
				bmu_loc = bmu_loc.squeeze()
				
				learning_rate_op = np.exp(-it/decay_rate)
				alpha_op = self.alpha * learning_rate_op
				sigma_op = self.sigma * learning_rate_op

				# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
				bmu_distance_squares = torch.sum(torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m*self.n)]).float(), 2), 1) # dim = som_dim = m*n
				neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
				learning_rate_op = alpha_op * neighbourhood_func

				learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1]*np.ones(self.input_data.dim) for i in range(self.m*self.n)]) # dim = (m*n, input_dim)
				delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.m*self.n)]) - self.weights))       # element-wise multiplication -> dim = (m*n, input_dim)
				new_weights = torch.add(self.weights, delta)
				self.weights = torch.nn.Parameter(new_weights)
				if wandb_log:
					image_grid=self.create_image_grid()
					# Create a figure and axis with a specific size
					fig, ax = plt.subplots(figsize=(image_grid.shape[1] / 10, image_grid.shape[0] / 10), dpi=500)
					ax.axis("off")
					ax.add_image(plt.imshow(image_grid))
					fig.canvas.draw()
					pil_image=PIL.Image.frombytes('RGB', 
						fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
					plt.close(fig)
					wandb.log({"weights": wandb.Image(pil_image)})

	def train_batch(self, dataset: datasets, batch_size: int, epochs: int, decay_rate: int, wandb_log: bool = False):
		"""
        Train the SOM using batch learning.

        Args:
            dataset (Dataset): Dataset for training.
            batch_size (int): Size of each batch.
            epochs (int): Number of epochs to train for.
			decay_rate (int): Decay rate for the learning rate.
        """
		print("\nSOM training with batch mode without pytorch optimizations is starting with hyper-parameters:")
		print("\u2022 batch size = "+str(batch_size))
		print("\u2022 epochs = "+str(epochs)+"\n\n")

		print("Creating a DataLoader object from dataset", end=" ",flush=True)
		data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713 \n",flush=True)

		for iter_no in tqdm(range(epochs), desc=f"Epoch"):
			for batch in data_loader:
				neighbourhood_func = self._neighbourhood_batch(batch[0],  decay_rate, iter_no)
				# update weights
				new_weights = torch.matmul(neighbourhood_func.T, batch[0]) # (som_dim, batch_size)x(batch_size, input_dim) = (som_dim, input_dim)
				norm = torch.sum(neighbourhood_func, 0) # som_dim
				self.weights = torch.nn.Parameter(torch.div(new_weights.T, norm).T)
			if wandb_log:
				image_grid=self.create_image_grid()
				# Create a figure and axis with a specific size
				fig, ax = plt.subplots(figsize=(image_grid.shape[1] / 10, image_grid.shape[0] / 10), dpi=500)
				ax.axis("off")
				ax.add_image(plt.imshow(image_grid))
				fig.canvas.draw()
				pil_image=PIL.Image.frombytes('RGB', 
					fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
				plt.close(fig)
				wandb.log({"weights": wandb.Image(pil_image)})
				
	def train_batch_pytorch(self, dataset: datasets, batch_size: int, epochs: int, learning_rate: float, decay_rate: int, wandb_log: bool = False):
		"""
		Train the SOM using PyTorch's built-in optimizers and backpropagation.

        Args:
            dataset (Dataset): Dataset for training.
            batch_size (int): Size of each batch.
            epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for the optimizer.
			decay_rate (int): Decay rate for the learning rate.
        """
		print("\nSTM training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		print("\u2022 batch size = "+str(batch_size))
		print("\u2022 epochs = "+str(epochs))
		print("\u2022 learning rate = "+str(learning_rate)+"\n\n")

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713 \n", flush=True)

		self.train()
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
		for iter_no in tqdm(range(epochs), desc=f"Epochs", leave=True, position=0):
			for b, batch in enumerate(data_loader):
				neighbourhood_func = self._neighbourhood_batch(batch[0], decay_rate, iter_no)
				distance_matrix = torch.cdist(batch[0], self.weights, p=2) # dim = (batch_size, som_dim) 
				loss = torch.mul(1/2,torch.sum(torch.mul(neighbourhood_func,distance_matrix)))

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				if wandb_log:
					image_grid=self.create_image_grid()
					# Create a figure and axis with a specific size
					fig, ax = plt.subplots(figsize=(image_grid.shape[1] / 10, image_grid.shape[0] / 10), dpi=500)
					ax.axis("off")
					ax.add_image(plt.imshow(image_grid))
					fig.canvas.draw()
					pil_image=PIL.Image.frombytes('RGB', 
						fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
					plt.close(fig)
					wandb.log({	
						"weights": wandb.Image(pil_image),
						"loss" : loss.item()
					})

		"""
		Create an image grid from the trained SOM weights.
		
		Args:
			som (SOM): The trained Self-Organizing Map.
		
		Returns:
			numpy array: heigh*width*channels array representing the image grid.
		"""
		image_grid = [[] for _ in range(self.m)]
		weights = self.get_weights()
		locations = self.get_locations()
		if self.input_data.type=="int":
			for i, loc in enumerate(locations):
				image_grid[loc[0]].append(weights[i].detach().numpy())
			return np.array(image_grid)
		else:
			# rearrange weight in a matrix called image_grid
			image_grid=torch.cat([torch.cat([self.input_data.inverse_transform_data(weights[i].detach()) for i in range(self.n)], 0) for j in range(self.m)], 1)
			return np.array(image_grid)