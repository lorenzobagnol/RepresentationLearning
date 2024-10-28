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

	def neighbourhood_batch(self, batch: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (Tuple[torch.Tensor]): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			decay_rate (int): Decay rate for the learning rate.
            it (int): Current iteration number.

        Returns:
            torch.Tensor: Neighborhood function values.
        """

		# look for the best matching unit (BMU)
		dists = batch.unsqueeze(1).expand((batch.shape[0], self.weights.shape[0], batch.shape[1])) - self.weights.unsqueeze(0).expand((batch.shape[0], self.weights.shape[0], batch.shape[1])) # (batch_size, som_dim, image_tot_dim)
		dists = torch.sum(torch.pow(dists,2), 2) # (batch_size, som_dim)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
		bmu_distances = self.locations.float() - bmu_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		bmu_distance_squares = torch.sum(torch.pow(bmu_distances, 2), 2) # (batch_size, som_dim)
		neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, radius**2))) # (batch_size, som_dim)
		return neighbourhood_func

