from typing import Sequence, Union, Tuple, Generator
import torch
import torch.nn as nn
import numpy as np
from abc import ABC

from utils.inputdata import InputData
from utils.config import SOMConfig

class SOM(nn.Module, ABC):
	"""
	Class of Self-Organizing Map.
	"""
	def __init__(self, m: int , n: int, input_data: InputData, sigma: float = None):
		"""
        Initialize the class for the SOM network.

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

		w=torch.rand(m*n, self.input_data.dim)
		self.weights = torch.nn.Parameter(torch.nn.init.xavier_normal_(w), requires_grad=True) #TODO Glorot initialization
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.locations = torch.LongTensor(np.array(list(self.neuron_locations()))).to(self.device)

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

	def neighbourhood_batch(self, dists: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            dists (torch.Tensor): Batch input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """

		# look for the best matching unit (BMU)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)
		return neighbourhood_func

	def forward(self, batch: torch.Tensor) -> torch.Tensor:
		"""
        Compute the distances for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch input vectors. B x D where D = total dimension (image_dim*channels)

        Returns:
            torch.Tensor: 
        """

		# look for the distances
		dists = batch.unsqueeze(1).expand((batch.shape[0], self.weights.shape[0], batch.shape[1])) - self.weights.unsqueeze(0).expand((batch.shape[0], self.weights.shape[0], batch.shape[1])) # (batch_size, som_dim, image_tot_dim)
		dists = torch.sum(torch.pow(dists,2), 2) # (batch_size, som_dim)

		return dists
	
	def _compute_gaussian(self, points: torch.Tensor, radius: float):
	
		distances = self.locations.float() - points.unsqueeze(1) # (batch_size, som_dim, 2)
		distance_squares = torch.sum(torch.pow(distances, 2), 2) # (batch_size, som_dim)
		gaussian_func = torch.exp(torch.neg(torch.div(distance_squares, radius**2))) # (batch_size, som_dim)
		return gaussian_func
