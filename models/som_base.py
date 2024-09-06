from typing import Sequence, Union, Tuple, Generator
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from utils.inputdata import InputData
from tqdm import tqdm
import wandb
from time import sleep
from abc import ABC, abstractmethod
import PIL

class BaseSOM(nn.Module, ABC):
	"""
	Base class of Self-Organizing Map.
	"""
	def __init__(self, m: int , n: int, input_data: InputData, sigma=None):
		"""
        Initialize the base class for the SOM network.

        Args:
            m (int): Number of rows in the SOM grid.
            n (int): Number of columns in the SOM grid.
            input_data (InputData): InputData object containing input dimensions.
            sigma (float, optional): Initial radius of the neighbourhood function. Defaults to half the maximum of m or n.
        """
		super().__init__()
		
		self.m = m
		self.n = n
		self.input_data = input_data
		if sigma is None:
			self.sigma = max(m, n) / 2.0
		else:
			self.sigma = float(sigma)

		self.weights = torch.nn.Parameter(torch.rand(m*n, self.input_data.dim))
		self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))

	def get_weights(self) -> torch.Tensor:
		return self.weights.detach()

	def get_locations(self) -> torch.LongTensor:
		return self.locations

	def neuron_locations(self):
		for i in range(self.m):
			for j in range(self.n):
				yield np.array([i, j])

	def map_vects(self, input_vects) -> list:
		to_return = []
		for vect in input_vects:
			min_index = min([i for i in range(len(self.weights))],
							key=lambda x: np.linalg.norm(vect-self.weights[x].detach()))
			to_return.append(self.locations[min_index])
		return to_return

	def _neighbourhood_batch(self, batch: torch.Tensor, decay_rate: int, it: int) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch of input vectors.
			decay_rate (int): Decay rate for the learning rate.
            it (int): Current iteration number.

        Returns:
            torch.Tensor: Neighborhood function values.
        """
		# look for the best matching unit (BMU)
		dists = torch.cdist(batch, self.weights, p=2) # (batch_size, som_dim)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 
		
		learning_rate_op = np.exp(-it/decay_rate)
		sigma_op = self.sigma * learning_rate_op

		# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
		bmu_distances = self.locations.float() - bmu_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		bmu_distance_squares = torch.sum(torch.pow(bmu_distances, 2), 2) # (batch_size, som_dim)
		neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2))) # (batch_size, som_dim)
		return neighbourhood_func

	def create_image_grid(self, clip_image: bool = None) -> np.ndarray:
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
		else:
			# rearrange weight in a matrix called image_grid
			image_grid=torch.cat([torch.cat([self.input_data.inverse_transform_data(weights[i].detach()) for i in range(self.n)], 0) for j in range(self.m)], 1)
		if clip_image:
			if self.input_data.channel_range=="RGB":
				return np.clip(image_grid, 0, 1)
			else:
				return np.int64(np.clip(image_grid, 0, 255))
		return np.array(image_grid)