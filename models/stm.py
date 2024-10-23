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


	def compute_local_competence(self, val_set: datasets, label: int, batch_size: int):

		self.eval()
		indices = torch.where(val_set.targets==label)[0].tolist()
		subset_val=torch.utils.data.Subset(val_set, indices)
		data_loader = torch.utils.data.DataLoader(subset_val,
										batch_size=batch_size,
										shuffle=False,
										)
		
		# compute mask around the target point
		target_loc=torch.stack([self.target_points[label] for i in range(batch_size)]) # (batch_size, 2) 
		target_distances = self.locations.float() - target_loc.unsqueeze(1)	# (batch_size, som_dim, 2)
		target_distances_squares = torch.sqrt+(torch.sum(torch.pow(target_distances, 2), 2)) # (batch_size, som_dim)
		mask = (target_distances_squares<self.sigma) 

		total_distance=0
		for b, batch in enumerate(data_loader):
			dists = batch[0].unsqueeze(1).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) - self.weights.unsqueeze(0).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) # (batch_size, som_dim, image_tot_dim)
			dists = torch.sum(torch.pow(dists,2), 2) # (batch_size, som_dim)
			masked_dists = dists*mask
			total_distance += torch.sum(masked_dists).item()
		
		total_distance /= len(subset_val)
		total_distance /= torch.sum(mask).item()

		self.train()
		return total_distance
			
	
	def target_distance_batch(self, batch: Tuple[torch.Tensor], radius: float) -> torch.Tensor:
		"""
		Computes the distance between the SOM (Self-Organizing Map) nodes and target points for a given batch of data.

		Args:
			batch (Tuple[torch.Tensor]): A batch with labeled data obtained from a DataLoader.
			radius (float): Variance of the gaussian.

		Returns:
            torch.Tensor: shape = (batch_size, som_dim) containing distances.
		"""
		target_loc=torch.stack([self.target_points[np.int32(batch[1])[i]] for i in range(batch[1].shape[0])]) # (batch_size, 2) 

		target_distances = self.locations.float() - target_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		target_distances_squares = torch.sum(torch.pow(target_distances, 2), 2) # (batch_size, som_dim)
		target_dist_func = torch.exp(torch.neg(torch.div(target_distances_squares, radius**2))) # (batch_size, som_dim)

		return target_dist_func


	def neighbourhood_batch_vieri(self, batch: Tuple[torch.Tensor], radius: float) -> torch.Tensor:
		"""
        Compute the neighborhood function for a batch of inputs.

        Args:
            batch (torch.Tensor): Batch of labeled input vectors. B x D where D = total dimension (image_dim*channels)
			radius (float): Variance of the gaussian.

        Returns:
            torch.Tensor: Neighborhood function values.
        """
		# look for the best matching unit (BMU)
		dists = batch[0].unsqueeze(1).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) - self.weights.unsqueeze(0).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) # (batch_size, som_dim, image_tot_dim)
		dists = torch.sum(torch.pow(dists,2), 2) # (batch_size, som_dim)
		# compute mask around the target point
		target_loc=torch.stack([self.target_points[int(label)] for label in batch[1]]) # (batch_size, 2) 
		target_distances = self.locations.float() - target_loc.unsqueeze(1)	# (batch_size, som_dim, 2)
		target_distances_squares = torch.sqrt(torch.sum(torch.pow(target_distances, 2), 2)) # (batch_size, som_dim)
		mask = torch.BoolTensor(target_distances_squares<radius) # (batch_size, som_dim)

		masked_distances = torch.where(mask, dists, torch.tensor(float('inf')))
		_, bmu_indices = torch.min(masked_distances, 1) # som_dim
		bmu_loc = torch.stack([self.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2) 


		# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
		bmu_distances = self.locations.float() - bmu_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		bmu_distance_squares = torch.sum(torch.pow(bmu_distances, 2), 2) # (batch_size, som_dim)
		neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, radius**2)))# (batch_size, som_dim)
		return neighbourhood_func

	
