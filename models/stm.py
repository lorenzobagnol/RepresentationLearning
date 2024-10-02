import io
import torch
import torch.nn as nn
import numpy as np
from matplotlib.figure import Figure
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

	def train_batch_pytorch(self, dataset_train: datasets, dataset_val: datasets, batch_size: int, epochs: int, learning_rate: float, decay_rate: float, wandb_log: bool = False, clip_images: bool = False):
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
		print("\u2022 decay_rate = "+str(decay_rate)+"\n\n\n")

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713", flush=True)
		print("\n\n\n")
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
				fig=self.resize_image_add_target_points(image_grid)
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

	def train_lifelong(self, dataset_train: datasets, dataset_val: datasets, batch_size: int, subset_size: int, epochs_per_subset: int, disjoint_training: bool, learning_rate: int, decay_rate: float, wandb_log: bool = False, clip_images: bool = False):
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
		print("\nSTM LifeLong learning with batch mode and pytorch optimizations is starting with hyper-parameters:")
		print("\u2022 batch size = "+str(batch_size))
		print("\u2022 subset_size = "+str(subset_size))
		print("\u2022 epochs_per_subset = "+str(epochs_per_subset))
		print("\u2022 subset disjoint = "+str(disjoint_training))
		print("\u2022 learning rate = "+str(learning_rate)+"\n\n\n")

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=batch_size,
                                          shuffle=True,)
		print("\u2713 \n", flush=True)
		print("\n\n\n")
		self.train()
		
		labels = list(set(dataset_train.targets.detach().tolist()))
		for i in range(len(labels)):
			if i not in labels:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		rep = math.ceil(len(labels)/subset_size)
		for i in range(rep):
			if i==1:
				self.sigma=1.5
				epochs_per_subset=5
				learning_rate/=10.
				print("Parameters changed:")
				print("sigma = "+str(self.sigma))
				print("learning_rate = "+str(learning_rate))
			print("Training on labels in range:\t"+str(i*subset_size) +" <= label < "+str((i+1)*subset_size))
			if disjoint_training:
				indices = torch.where((dataset_train.targets>=i*subset_size) & (dataset_train.targets<(i+1)*subset_size))[0].tolist()
			else:
				indices = torch.where(dataset_train.targets<(i+1)*subset_size)[0].tolist()

			subset_lll=torch.utils.data.Subset(dataset_train, indices)
			print("This subset contains "+str(len(subset_lll))+" elements.")
			data_loader = torch.utils.data.DataLoader(subset_lll,
											batch_size=batch_size,
											shuffle=True,
											)
			decay_rate=int((len(subset_lll)/batch_size)*epochs_per_subset) # network is very instable wrt this parameter. do not touch
			optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
			stop_flag = False
			for iter_no in tqdm(range(epochs_per_subset), desc=f"Epochs", leave=True, position=0):
				for b, batch in enumerate(data_loader):
					
					neighbourhood_func = self._neighbourhood_batch(batch[0], decay_rate, b+(iter_no*len(subset_lll)/batch_size)) 
					target_dist = self._target_distance_batch(batch, decay_rate, b+(iter_no*len(subset_lll)/batch_size))
					weight_function = torch.mul(neighbourhood_func, target_dist)
					distance_matrix = batch[0].unsqueeze(1).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) - self.weights.unsqueeze(0).expand((batch[0].shape[0], self.weights.shape[0], batch[0].shape[1])) # dim = (batch_size, som_dim, input_dim) 
					distance_matrix_norms = torch.sqrt(torch.sum(torch.pow(distance_matrix,2), 2)) # dim = (batch_size, som_dim) 
					loss = torch.mul(1/2,torch.sum(torch.mul(weight_function, distance_matrix_norms)))

					loss.backward()
					optimizer.step()
					optimizer.zero_grad()
					
				# with torch.no_grad():
				# 	local_competence=self._compute_local_competence(val_set=dataset_val, label=i, batch_size=batch_size)
				# 	print(local_competence)
					
				# if local_competence < 0.0001:
				# 	print("Training interrupted for this subest after "+str(iter_no)+" epochs.")
				# 	print("Small local competence obtained : "+str(local_competence)+"\n")
				# 	stop_flag = True
				# 	break

				if wandb_log:
					image_grid=self.create_image_grid(clip_images)
					fig=self.resize_image_add_target_points(image_grid)
					fig.canvas.draw()
					pil_image=PIL.Image.frombytes('RGB', 
						fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
					plt.close(fig)
					wandb.log({	
						"weights": wandb.Image(pil_image),
						"loss" : loss.item(),
						#"local competence" : local_competence
					})
		if wandb.run is not None:
			wandb.finish()
		return
	
	def _compute_local_competence(self, val_set: datasets, label: int, batch_size: int):

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
		target_distances_squares = torch.sqrt(torch.sum(torch.pow(target_distances, 2), 2)) # (batch_size, som_dim)
		mask = (target_distances_squares<self.sigma) #TODO see if it makes sense with small sigma around 1.5

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
		# sigma_op= 5 #TODO: move Hyperparameter outside the function
		# sigma_op = sigma_op * learning_rate_op

		target_distances = self.locations.float() - target_loc.unsqueeze(1) # (batch_size, som_dim, 2)
		target_distances_squares = torch.sum(torch.pow(target_distances, 2), 2) # (batch_size, som_dim)
		target_dist_func = torch.exp(torch.neg(torch.div(target_distances_squares, sigma_op**2))) # (batch_size, som_dim)

		return target_dist_func


	def resize_image_add_target_points(self, image_grid: np.ndarray) -> Figure:
		target_width = 800  
		target_height = 800  
		dpi_value = min(300, max(72, target_width / image_grid.shape[1]))
		figsize_x = target_width / dpi_value
		figsize_y = target_height / dpi_value
		fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi_value)
		base_font_size = 24  
		font_size = base_font_size * (dpi_value / 100)  
		if self.input_data.type=="int":
			for key, value in self.target_points.items():
				ax.text(value[0], value[1], str(key), ha='center', va='center',
					bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=font_size)
		else:
			for key, value in self.target_points.items():
				ax.text(value[0]*self.input_data.dim1, value[1]*self.input_data.dim2, str(key), ha='center', va='center',
					bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=font_size)
		ax.imshow(image_grid)
		ax.axis("off")

		return fig





#TODO: we are using sigma, not sigma^2
