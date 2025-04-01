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
	def __init__(self, m: int , n: int, input_data: InputData):
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

		w=torch.rand(m*n, self.input_data.dim)
		self.weights = torch.nn.Parameter(1e-4*torch.nn.init.xavier_normal_(w), requires_grad=True) #TODO verify
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
	
	
	def find_bmu(self, dists: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute the best matching unit (BMU) for a batch of inputs.

		Args:
			dists (torch.Tensor): Batch input vectors. B x D where D = total dimension (image_dim*channels)
		
		Returns:
			bmu (torch.Tensor): The best matching unit (BMU) for each input vector.
			bmu_loc (torch.Tensor): The locations of the BMUs in the SOM grid.
        """

		# look for the best matching unit (BMU)
		min_dist, bmu_indices = torch.min(dists, 1) # som_dim
		bmu = self.weights[bmu_indices] # (batch_size, image_tot_dim)
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		return bmu, bmu_loc
	

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
		dists_norm_sq = torch.sum(torch.pow(dists,2), 2) # (batch_size, som_dim)

		return dists_norm_sq






class SOMLoss:

	def __init__(self, model: SOM, device: torch.device, sigma: float = None):

		self.device = device
		self.model = model
		self.weight_function = lambda **kwargs: (
						self.neighbourhood_batch(**kwargs)
						)


	def __call__(self, norm_distance_matrix: torch.Tensor, sigma_local: float) -> torch.Tensor:

		weight_function = self.weight_function(
			dists=norm_distance_matrix, 
			radius=sigma_local
			)	
		loss = torch.mul(1/2,torch.sum(torch.mul(weight_function, norm_distance_matrix)))

		return loss


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
		bmu, bmu_loc = self.model.find_bmu(dists) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)
		return neighbourhood_func
	

	def _compute_gaussian(self, points: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute a normalized gaussian centered in a batch of points with a certain radius.

        """

		distances = self.model.locations.float() - points.unsqueeze(1) # (batch_size, som_dim, 2)
		distance_squares = torch.sum(torch.pow(distances, 2), 2) # (batch_size, som_dim)
		gaussian_func = torch.exp(torch.neg(torch.div(distance_squares, radius**2))) # (batch_size, som_dim)
		return gaussian_func
	
	
	def _compute_tanh(self, points: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute an hyperbolic tangent function centered in a batch of points with a certain radius.

        """
		
		distances = self.model.locations.float() - points.unsqueeze(1) # (batch_size, som_dim, 2)
		distance_squares = torch.sum(torch.pow(distances, 2), 2) # (batch_size, som_dim)
		tanh_weight_function = torch.tanh(torch.div((radius**2),distance_squares))   
		return tanh_weight_function
