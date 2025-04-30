from typing import List, Literal, Set, Tuple, Union
import torch
import numpy as np
from torch import Tensor
import math
import random

from utils.inputdata import InputData
from models.som import SOM


class TargetPoint:

	def __init__(self, value: Tensor, usable: bool = True, label: int = None):
		self.value = value
		self.usable = usable
		self.label = label


class TargetPoints:

	def __init__(self, num_init_points: int, device: torch.device, M: int, N: int, seed: int = None):
		self.seed = seed
		self.num_points = num_init_points
		self.device = device
		self.M = M
		self.N = N
		self.points = self.generate_equally_distributed_points_v2(shuffle=True)

	def find_nearest_point(self, point: Tensor, available:bool, top_k: int=1) -> Union[List[TargetPoint], TargetPoint]:
		"""
		Find the nearest target point to a given point.

		Args:
			point (Tensor): The point to find the nearest target point to.
			top_k (int, optional): The number of nearest points to return. Defaults to 1.

		Returns:
			The nearest(s) target point.
		"""
		if available:
			available_points = [p for p in self.points if p.usable]
		else:
			available_points = [p for p in self.points]

		if len(available_points) == 0:
			return None

		dists = torch.norm(point - torch.stack([p.value for p in available_points]), dim=1)
		nearest_indices = torch.topk(dists, top_k, largest=False).indices
		if top_k == 1:
			return available_points[nearest_indices[0]]
		else:
			return [available_points[i] for i in nearest_indices]
	
	def generate_equally_distributed_points(self) -> Set[TargetPoint]:
		m=self.M
		n=self.N
		# Adjust M and N to exclude the borders (from 1 to M-2 and 1 to N-2)
		if m <= 2 or n <= 2:
			raise ValueError("Grid is too small to exclude borders.")
		# Compute the best possible factors for k_x and k_y, excluding borders
		k_x = int(math.sqrt(self.num_points * (m - 2) / (n - 2)))  # Approximate number of points in the x (row) direction
		k_y = int(math.sqrt(self.num_points * (n - 2) / (m - 2)))  # Approximate number of points inself.config. the y (col) direction
		# Adjust k_x and k_y to ensure the total number of points is at least self.num_points
		while k_x * k_y < self.num_points:
			if k_x < k_y:
				k_x += 1
			else:
				k_y += 1
		# Compute step sizes in the inner grid (excluding borders)
		step_x = (m - 2 - 1) / (k_x - 1) if k_x > 1 else 0
		step_y = (n - 2 - 1) / (k_y - 1) if k_y > 1 else 0
		# Generate points in the range [1, M-2] and [1, N-2] to avoid borders
		x_coords = [round(1 + i * step_x) for i in range(k_x)]
		y_coords = [round(1 + j * step_y) for j in range(k_y)]
		# Combine x and y coordinates to get the points
		points = [(x, y) for x in x_coords for y in y_coords]
		target_points=set(TargetPoint(torch.Tensor(v).to(self.device)) for k,v in enumerate(points[:self.num_points]))
		return target_points
	

	def generate_equally_distributed_points_v2(self, shuffle:bool=False) -> Set[TargetPoint]:
		points = np.array(
				[
					[0.15, 0.17],
					[0.12, 0.54],
					[0.16, 0.84],
					[0.50, 0.15],
					[0.36, 0.45],
					[0.62, 0.50],
					[0.48, 0.82],
					[0.83, 0.17],
					[0.88, 0.50],
					[0.83, 0.83],
				]
			)
		points_list=np.int32(points*min(self.M, self.N)).tolist()
		if self.seed is not None:
			random.seed(self.seed)
		index_list = list(range(len(points_list)))
		random.shuffle(index_list)
		target_points=set(TargetPoint(torch.Tensor(v).to(self.device), True, k) for k,v in zip(index_list, points_list))
		return target_points
	

	def generate_new_point(self, random:bool, label:int=None) -> TargetPoint:

		"""
		Generate a new target point.

		Args:
			random (bool): Whether to generate a random point or not.
			label (int, optional): The label of the point. Defaults to None.

		Returns:
			TargetPoint: The generated target point.
		"""
		pass
		self.num_points += 1
		if random:
			return 
		else:
			return
		
		
	def get_point_from_label(self, label: int) -> TargetPoint:
		"""
		Get a target point from a given label.

		Args:
			label (int): The label of the target point.

		Returns:
			TargetPoint: The target point with the given label.
		"""
		return next(target_point for target_point in self.points if target_point.label == label)


class STMLoss:

	def __init__(self, model: SOM, device: torch.device, mode: Literal["STC-modified", "Base", "Base-STC", "Base_Norm", "BGN"], target_points: TargetPoints):

		self.device = device
		self.model = model
		self.target_points = target_points	
		
		match mode:

			case "STC":

				self.weight_function = lambda dists, **kwargs: self.neighbourhood_batch_vieri_modified(dists, kwargs["radius"], kwargs["labels"])

			case "Base":

				self.weight_function = lambda dists, **kwargs: (
					torch.mul(
						self.neighbourhood_batch(dists, kwargs["radius"]),
						self.target_distance_batch(kwargs["labels"], kwargs["radius"])
					)
				)

			case "BGN": # not working. Learn where it shouldn't 

				def BGN(dists, **kwargs):
					radius = kwargs["radius"]
					labels = kwargs["labels"]
					weight_function = self.gaussian_product_normalizer(dists, radius, labels)
					return weight_function
				
				self.weight_function = BGN

			case "Base_Norm": # not working. Often divides by zero

				def Base_Norm(dists, **kwargs):
					sigma_local = kwargs["sigma_local"]
					target_radius = kwargs["TARGET_RADIUS"]
					labels = kwargs["labels"]
					neighbourhood_func = self.neighbourhood_batch(dists, radius=sigma_local)
					target_dist = self.target_distance_batch(labels, radius=target_radius)
					weight_function = torch.mul(neighbourhood_func, target_dist)
					max_weight_function = torch.max(weight_function,1).values
					if torch.min(max_weight_function)==0:
						print("loss normalization contains zeros.")
						return
					weight_function = torch.div(weight_function, max_weight_function.unsqueeze(1))

				self.weight_function = Base_Norm

			case "Base-STC":

				self.weight_function = lambda dists, **kwargs: self.hybrid_weight_function(dists, kwargs["radius"], kwargs["labels"])

			case "STC-modified":

				self.weight_function = lambda dists, **kwargs: self.neighbourhood_batch_vieri_modified(dists, kwargs["radius"], kwargs["labels"])


	def __call__(self, som_output: torch.Tensor, labels, sigma_local: float, target_radius: float) -> torch.Tensor:

		weight_function = self.weight_function(
			dists=som_output, 
			labels=labels, 
			radius=sigma_local,
			target_radius=target_radius
			)	
		loss = torch.mul(1/2,torch.sum(torch.mul(weight_function, som_output)))
		return loss
		
	def target_distance_batch(self, labels, radius: float) -> torch.Tensor:
		"""
		Compute the target distance function for a batch of inputs.

		Args:
			batch (torch.Tensor): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

		Returns:
			torch.Tensor: Target distance function values.
		"""
		target_loc = torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 

		target_dist_func = self._compute_gaussian(target_loc, radius) # (batch_size, som_dim)

		return target_dist_func


	def neighbourhood_batch_vieri(self, dists: torch.Tensor, radius: float, labels) -> torch.Tensor:
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
		target_loc = torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 
		target_distances = self.model.locations.float() - target_loc.unsqueeze(1)	# (batch_size, som_dim, 2)
		target_distances_squares = torch.sqrt(torch.sum(torch.pow(target_distances, 2), 2)) # (batch_size, som_dim)
		mask = (target_distances_squares<radius).to(self.device) # (batch_size, som_dim)

		masked_distances = torch.where(mask, dists, torch.tensor(float('inf'))) # (batch_size, som_dim)
		_, bmu_indices = torch.min(masked_distances, 1) # batch_size
		bmu_loc = torch.stack([self.model.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)	
		return neighbourhood_func


	def target_and_bmu_weighted_batch(self, dists: torch.Tensor, radius: float, labels) -> torch.Tensor:
		"""
		Computes the gaussian centered in an average points of target and BMU

		Args:
			batch (torch.Tensor): A batch with data obtained from a DataLoader.
			radius (float): Variance of the gaussian.

		Returns:
            torch.Tensor: shape = (batch_size, som_dim) containing distances.
		"""
		target_loc=torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 

		# look for the best matching unit (BMU)
		bmu, bmu_loc = self.model.find_bmu(dists) # (batch_size, 2) 

		distances_targets_bmus = torch.sum(torch.pow(target_loc-bmu_loc,2),1)
		max_distance=np.square(10)+np.square(30)
		normalized_distances_targets_bmus = (distances_targets_bmus/max_distance).unsqueeze(1)

		average_points = normalized_distances_targets_bmus*target_loc + (1-normalized_distances_targets_bmus)*bmu_loc # (batch_size, 2) 
		average_dist_func = self._compute_gaussian(average_points, radius) # (batch_size, som_dim)

		return average_dist_func
	

	def hybrid_weight_function(self, dists: torch.Tensor, radius: float, labels) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            dists (torch.Tensor): Norm squared of distance batch-weights (batch_size, som_dim).
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """
		# look for the best matching unit (BMU)
		bmu, bmu_loc = self.model.find_bmu(dists) # (batch_size, 2) 
		# compute target points
		target_loc = torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 
		# compute distance from target points and BMUs in the batch
		bmu_target_distances = torch.sqrt(torch.sum(torch.pow(bmu_loc-target_loc,2), 1)) # (batch_size)

		if (torch.max(bmu_target_distances)>5.): # Vieri mode
			hybrid_weight_function = self.neighbourhood_batch_vieri(dists, labels, radius=radius)
		else:  # base mode
			neighbourhood_func = self.neighbourhood_batch(dists, radius=radius)
			target_dist = self.target_distance_batch(labels, radius=radius)
			hybrid_weight_function = torch.mul(neighbourhood_func, target_dist)

		return hybrid_weight_function
	

	def neighbourhood_batch_vieri_modified(self, dists: torch.Tensor, radius: float, labels) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """

		target_loc = torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 
		weighted_dists = torch.mul(dists, torch.neg(self._compute_gaussian(target_loc, radius)))
		_, bmu_indices = torch.min(weighted_dists, 1) # som_dim
		bmu_loc = torch.stack([self.model.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 

		neighbourhood_func = self._compute_gaussian(bmu_loc, radius) # (batch_size, som_dim)
		
		return neighbourhood_func
	

	def gaussian_product_normalizer(self, dists: torch.Tensor, radius: float, labels) -> torch.Tensor:

		target_loc = torch.stack([self.target_points.get_point_from_label(int(label)).value for label in labels]) # (batch_size, 2) 

		# look for the best matching unit (BMU)
		bmu, bmu_loc = self.model.find_bmu(dists) # (batch_size, 2) 

		average_points = torch.div(target_loc + bmu_loc, 2.) # (batch_size, 2) 

		distance_squares = torch.sum(torch.pow(torch.div(average_points-target_loc,2), 2), 1) # (batch_size, som_dim)

		# |average_points-target_loc|=|average_points-bmu_loc|
		maximum_value = torch.exp(-torch.div(distance_squares+distance_squares, radius**2)) # (batch_size) 

		return maximum_value 
	

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

	


		
