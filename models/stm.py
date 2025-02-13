from typing import Tuple
import torch
import numpy as np
from torchvision import datasets

from utils.inputdata import InputData
from models.som import SOM
 

class STM(SOM):
	"""
	Supervised-Topological Map with Gaussian Neighbourhood function and target function.
	"""
	def __init__(self, m: int , n: int, input_data: InputData, target_points: dict, sigma=None):
		"""
        Initialize the STM.

        Args:
            m (int): Number of rows in the SOM grid.
            n (int): Number of columns in the SOM grid.
            input_data (InputData): InputData object containing input dimensions.
			target_points (dict): Key are labels and values are torch Tensor indicating a point in the grid.
            sigma (float, optional): Initial radius of the neighbourhood function. Defaults to half the maximum of m or n.
        """
		super().__init__(m = m, n = n, input_data = input_data, sigma = sigma)
		self.target_points = target_points

			
	
	def target_distance_batch(self, targets: torch.Tensor, radius: float) -> torch.Tensor:
		"""
		Computes the distance between the SOM (Self-Organizing Map) nodes and target points for a given batch of data.

		Args:
			batch (torch.Tensor): A batch with data obtained from a DataLoader.
			radius (float): Variance of the gaussian.

		Returns:
            torch.Tensor: shape = (batch_size, som_dim) containing distances.
		"""
		target_loc = torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 

		target_dist_func = self._compute_gaussian(target_loc, radius) # (batch_size, som_dim)

		return target_dist_func


	def neighbourhood_batch_vieri(self, dists: torch.Tensor, targets: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """
		# look for the best matching unit (BMU)
		# compute mask around the target point
		target_loc = torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 
		target_distances = self.locations.float() - target_loc.unsqueeze(1)	# (batch_size, som_dim, 2)
		target_distances_squares = torch.sqrt(torch.sum(torch.pow(target_distances, 2), 2)) # (batch_size, som_dim)
		mask = (target_distances_squares<radius).to(self.device) # (batch_size, som_dim)

		masked_distances = torch.where(mask, dists, torch.tensor(float('inf'))) # (batch_size, som_dim)
		_, bmu_indices = torch.min(masked_distances, 1) # batch_size
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)	
		return neighbourhood_func


	def target_and_bmu_weighted_batch(self, dists: torch.Tensor, targets: torch.Tensor, radius: float) -> torch.Tensor:
		"""
		Computes the gaussian centered in an average points of target and BMU

		Args:
			batch (torch.Tensor): A batch with data obtained from a DataLoader.
			radius (float): Variance of the gaussian.

		Returns:
            torch.Tensor: shape = (batch_size, som_dim) containing distances.
		"""
		target_loc=torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 

		# look for the best matching unit (BMU)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		distances_targets_bmus = torch.sum(torch.pow(target_loc-bmu_loc,2),1)
		max_distance=np.square(10)+np.square(30)
		normalized_distances_targets_bmus = (distances_targets_bmus/max_distance).unsqueeze(1)

		average_points = normalized_distances_targets_bmus*target_loc + (1-normalized_distances_targets_bmus)*bmu_loc # (batch_size, 2) 
		average_dist_func = self._compute_gaussian(average_points, radius) # (batch_size, som_dim)

		return average_dist_func

	def hybrid_weight_function(self, dists: torch.Tensor, targets: torch.Tensor, radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            dists (torch.Tensor): Norm squared of distance batch-weights (batch_size, som_dim).
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """
		# look for the best matching unit (BMU)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 
		# compute target points
		target_loc=torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 
		# compute distance from target points and BMUs in the batch
		bmu_target_distances = torch.sqrt(torch.sum(torch.pow(bmu_loc-target_loc,2), 1)) # (batch_size)

		if (torch.max(bmu_target_distances)>5.): # Vieri mode
			hybrid_weight_function = self.neighbourhood_batch_vieri(dists, targets, radius=radius)
		else:  # base mode
			neighbourhood_func = self.neighbourhood_batch(dists, radius=radius)
			target_dist = self.target_distance_batch(targets, radius=radius)
			hybrid_weight_function = torch.mul(neighbourhood_func, target_dist)

		return hybrid_weight_function
	

	def neighbourhood_batch_vieri_modified(self, dists: torch.Tensor, targets: torch.Tensor, radius: float, target_radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """

		target_loc=torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 
		weighted_dists = torch.mul(dists, torch.neg(self._compute_gaussian(target_loc, radius)))
		_, bmu_indices = torch.min(weighted_dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)
		
		return neighbourhood_func

	def gaussian_product_normalizer(self, dists: torch.Tensor, targets: torch.Tensor, radius: float) -> torch.Tensor:

		target_loc=torch.stack([self.target_points[int(label)] for label in targets]) # (batch_size, 2) 

		# look for the best matching unit (BMU)
		_, bmu_indices = torch.min(dists, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		average_points = torch.div(target_loc + bmu_loc, 2.) # (batch_size, 2) 

		distance_squares = torch.sum(torch.pow(torch.div(average_points-target_loc,2), 2), 1) # (batch_size, som_dim)

		# |average_points-target_loc|=|average_points-bmu_loc|
		maximum_value = torch.exp(-torch.div(distance_squares+distance_squares, radius**2)) # (batch_size) 

		return maximum_value 
	
