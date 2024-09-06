from typing import Sequence, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import PIL
from torchvision import datasets
from utils.inputdata import InputData
from models.som_base import BaseSOM
from tqdm import tqdm
import wandb
import math
 

class STM(BaseSOM):
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

	def train_batch_pytorch(self, dataset: datasets, batch_size: int, epochs: int, learning_rate: float, decay_rate: float, wandb_log: bool = False, clip_images: bool = False):
		"""
		Train the STM using PyTorch's built-in optimizers and backpropagation.

        Args:
            dataset (Dataset): Dataset for training.
            batch_size (int): Size of each batch.
            epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for the optimizer.
			decay_rate (float): Decay rate for the learning rate.
        """
		print("\nSTM training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		print("\u2022 batch size = "+str(batch_size))
		print("\u2022 epochs = "+str(epochs))
		print("\u2022 learning rate = "+str(learning_rate))
		print("\u2022 decay_rate = "+str(decay_rate))

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713", flush=True)

		self.train()
		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
		for iter_no in tqdm(range(epochs), desc=f"Epochs", leave=True, position=0):
			for b, batch in enumerate(data_loader):
				neighbourhood_func = self._neighbourhood_batch(batch[0], decay_rate, iter_no)
				target_dist = self._target_distance_batch(batch, decay_rate, iter_no)
				weight_function = torch.mul(neighbourhood_func, target_dist)
				distance_matrix = torch.cdist(batch[0], self.weights, p=2) # dim = (batch_size, som_dim) 
				loss = torch.mul(1/2,torch.sum(torch.mul(weight_function,distance_matrix)))

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				if wandb_log:
					image_grid=self.create_image_grid(clip_images)
					# Create a figure and axis with a specific size
					fig, ax = plt.subplots(figsize=(image_grid.shape[1] / 10, image_grid.shape[0] / 10), dpi=500)
					ax.axis("off")
					ax.add_image(plt.imshow(image_grid))
					for key, value in self.target_points.items():
						plt.text(value[1], value[0], str(key), ha='center', va='center',
             				bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=4)
					fig.canvas.draw()
					pil_image=PIL.Image.frombytes('RGB', 
						fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
					plt.close(fig)
					wandb.log({	
						"weights": wandb.Image(pil_image),
						"loss" : loss.item()
					})
		if wandb.run is not None:
			wandb.finish()
		return

	def train_lifelong(self, dataset: datasets, batch_size: int, subset_size: int, epochs_per_subset: int, disjoint_training: bool, learning_rate: int, decay_rate: float, wandb_log: bool = False, clip_images: bool = False):
		"""
		Train the STM using a Continual Learning approach. The dataset is divided basing on labels and the training is divided too. PyTorch's built-in optimizers and backpropagation.

        Args:
            dataset (Dataset): Dataset for training.
            batch_size (int): Size of each batch.
			subset_size (int): Number of labels considered for each repetition of the Continual Learning process.
			epochs_per_subset (int): Number of epochs to train for each subset of the dataset.
			disjoint_training (bool): If true, subsets of the dataset are disjoint. If false each subset contains the previous one.
            learning_rate (float): Learning rate for the optimizer.
			decay_rate (float): Decay rate for the learning rate.
        """
		print("\nSTM training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		print("\u2022 batch size = "+str(batch_size))
		print("\u2022 subset_size = "+str(subset_size))
		print("\u2022 epochs_per_subset = "+str(epochs_per_subset))
		print("\u2022 subset disjoint = "+str(disjoint_training))
		print("\u2022 learning rate = "+str(learning_rate))
		print("\u2022 decay_rate = "+str(decay_rate))

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713 \n", flush=True)
		
		self.train()
		
		labels = list(set(dataset.targets.detach().tolist()))
		for i in range(len(labels)):
			if i not in labels:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		rep = math.ceil(len(labels)/subset_size)
		#counter = {lab : 0 for lab in labels}
		for i in range(rep):
			print("Training on labels in range "+str(i*subset_size) +"<"+str((i+1)*subset_size))
			if disjoint_training:
				indices = torch.where((dataset.targets>=i*subset_size) & (dataset.targets<(i+1)*subset_size))[0].tolist()
			else:
				indices = torch.where(dataset.targets<(i+1)*subset_size)[0].tolist()

			data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
											batch_size=batch_size,
											shuffle=True,
											)
			
			optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
			for iter_no in tqdm(range(epochs_per_subset), desc=f"Epochs", leave=True, position=0):
				for b, batch in enumerate(data_loader):
					neighbourhood_func = self._neighbourhood_batch(batch[0], decay_rate, iter_no)
					target_dist = self._target_distance_batch(batch, decay_rate, iter_no)
					weight_function = torch.mul(neighbourhood_func, target_dist)
					distance_matrix = torch.cdist(batch[0], self.weights, p=2) # dim = (batch_size, som_dim) 
					loss = torch.mul(1/2,torch.sum(torch.mul(weight_function,distance_matrix)))

					loss.backward()
					optimizer.step()
					optimizer.zero_grad()

					# self._update_counter(counter) come tengo conto delle posizioni dove la rete ha già imparato usando solo il counter?

					if wandb_log:
						image_grid=self.create_image_grid(clip_images)
						dists = torch.cdist(batch[0], self.weights, p=2) # (batch_size, som_dim)
						_, bmu_indices = torch.min(dists, 1)
						# Create a figure and axis with a specific size
						fig, ax = plt.subplots(figsize=(image_grid.shape[1] / 10, image_grid.shape[0] / 10), dpi=500)
						ax.axis("off")
						ax.add_image(plt.imshow(image_grid))
						for key, value in self.target_points.items():
							plt.text(value[1], value[0], str(key), ha='center', va='center',
								bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=4)
						#TODO: delete BMU plot
						plt.text(math.floor(bmu_indices/self.m), (bmu_indices%self.m), "BMU", ha='center', va='center',
             			bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=11)
						fig.canvas.draw()
						pil_image=PIL.Image.frombytes('RGB', 
							fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
						plt.close(fig)
						wandb.log({	
							"weights": wandb.Image(pil_image),
							"loss" : loss.item()
						})
		if wandb.run is not None:
			wandb.finish()
		return
	
	def _target_distance_batch(self, batch: torch.Tensor, decay_rate: int, it: int) -> torch.Tensor:
		"""
		Computes the distance between the SOM (Self-Organizing Map) nodes and target points for a given batch of data.

		Args:
			batch (torch.Tensor): A batch with labeled data obtained from a DataLoader.
			it (int) Current iteration number.

		Returns:
            torch.Tensor: shape = (batch_size, som_dim) containing distances.
		"""
		target_loc=torch.stack([self.target_points[np.int32(batch[1])[i]] for i in range(batch[1].shape[0])]) # (batch_size, 2) 
		
		learning_rate_op = np.exp(-it/decay_rate)
		sigma_op = self.sigma * learning_rate_op

		target_distances = self.locations.float() - target_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		target_distances_squares = torch.sum(torch.pow(target_distances, 2), 2) # (batch_size, som_dim)
		target_dist_func = torch.exp(torch.neg(torch.div(target_distances_squares, sigma_op**2))) # (batch_size, som_dim)

		return target_dist_func

	def _update_counter(self, dictionary: dict, increment_list: list):
		for el in increment_list:
			dictionary[el]+=1