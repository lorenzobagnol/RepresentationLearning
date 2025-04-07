import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import math
import random
import matplotlib.pyplot as plt
from typing import Literal, Sequence, Union

from utils.plotter import SOMPlotter
from models.som import SOM, SOMLoss


class SOMTrainer():
	

	def __init__(self, model: SOM, device, wandb_log: bool, clip_images: bool = False):

		self.model = model
		self.wandb_log = wandb_log
		self.clip_images = clip_images
		self.device = device


	def train_online(self, dataset_train: Dataset, dataset_val: Dataset, **kwargs):
		"""
		Train the SOM using online learning.

		Args:
			dataset_train (Dataset): Dataset for training.
			dataset_val (Dataset): The validation dataset used to evaluate the model's competence after training.
			**kwargs: Keyword arguments for various training hyperparameters, including:
				
		Returns:
			None: This function does not return any values, but it updates the model's weights and logs progress.
		
		"""
		print("\nSOM online-training is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		if self.wandb_log:
			wandb.config.update(kwargs)
			plotter = SOMPlotter(self.model, self.clip_images)
		
		for it in range(kwargs["EPOCHS"]):
			for i, el in tqdm(enumerate(dataset_train), f"epoch {it+1}", len(dataset_train)):
				x=el[0]
				# look for the best matching unit (BMU)
				dists = torch.pairwise_distance(x, self.model.weights, p=2)
				_, bmu_index = torch.min(dists, 0)
				bmu_loc = self.model.locations[bmu_index,:]
				bmu_loc = bmu_loc.squeeze()
				
				learning_rate_op = np.exp(-it/kwargs["EPOCHS"])
				sigma_op = kwargs["SIGMA"] * learning_rate_op

				# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
				bmu_distance_squares = torch.sum(torch.pow(self.model.locations.float() - bmu_loc.unsqueeze(0).float(), 2), 1) # dim = self.model_dim = m*n
				neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
				neighbourhood_func_multiplier = torch.stack([neighbourhood_func[i:i+1]*np.ones(self.model.input_data.dim) for i in range(self.model.m*self.model.n)]) # dim = (m*n, input_dim)
				
				delta = torch.mul(neighbourhood_func_multiplier, x.unsqueeze(0) - self.model.weights)       # element-wise multiplication -> dim = (m*n, input_dim)
				new_weights = torch.add(self.model.weights, delta)
				self.model.weights = torch.nn.Parameter(new_weights)
				if self.wandb_log:
					pil_image = plotter.create_pil_image()
					wandb.log({"weights": wandb.Image(pil_image)})
		if wandb.run is not None:
			wandb.finish()
		return


	def train_simple_batch(self,  dataset_train: Dataset, dataset_val: Dataset, **kwargs):
		"""
		Train the SOM using batch learning.

		Args:
			dataset_train (Dataset): Dataset for training.
			dataset_val (Dataset): The validation dataset used to evaluate the model's competence after training.
			**kwargs: Keyword arguments for various training hyperparameters, including:
				
		Returns:
			None: This function does not return any values, but it updates the model's weights and logs progress.
		
		"""
		print("\nSOM training with batch mode without pytorch optimizations is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		if self.wandb_log:
			wandb.config.update(kwargs)
			plotter = SOMPlotter(self.model, self.clip_images) 

		print("Creating a DataLoader object from dataset", end=" ",flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,)
		print("\u2713 \n",flush=True)

		som_loss = SOMLoss(self.model, self.device)
		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epoch"):
			sigma_local = kwargs["SIGMA"]*math.exp(-kwargs["BETA"]*iter_no)
			for batch in data_loader:
				inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
				norm_distance_matrix = self.model(inputs)
				neighbourhood_func = som_loss.neighbourhood_batch(norm_distance_matrix, sigma_local)
				# update weights
				new_weights = torch.matmul(neighbourhood_func.T, batch[0]) # (som_dim, batch_size)x(batch_size, input_dim) = (som_dim, input_dim)
				norm = torch.sum(neighbourhood_func, 0) # som_dim
				self.weights = torch.nn.Parameter(torch.div(new_weights.T, norm).T)
			if self.wandb_log:
				pil_image = plotter.create_pil_image()
				wandb.log({"weights": wandb.Image(pil_image)})
		if wandb.run is not None:
			wandb.finish()
		return


	def train_pytorch_batch(self, dataset_train: Dataset, dataset_val: Dataset, **kwargs):
		"""
		Train the SOM using PyTorch's built-in optimizers and backpropagation.

		Args:
			dataset_train (Dataset): Dataset for training.
			dataset_val (Dataset): The validation dataset used to evaluate the model's competence after training.
			**kwargs: Keyword arguments for various training hyperparameters, including:
				
		Returns:
			None: This function does not return any values, but it updates the model's weights and logs progress.
		
		"""
		print("\nSOM training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		if self.wandb_log:
			wandb.config.update(kwargs)
			plotter = SOMPlotter(self.model, self.clip_images)


		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											drop_last=True)
		print("\u2713", flush=True)
		print("\n\n\n")
		self.model.train()
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		som_loss = SOMLoss(self.model, self.device)
		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epochs", leave=True, position=0):
			lr_local = math.exp(-kwargs["BETA"]*iter_no)
			sigma_local = kwargs["SIGMA"]*math.exp(-kwargs["BETA"]*iter_no)
			for b, batch in enumerate(data_loader):
				inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
				norm_distance_matrix = self.model(inputs)
				loss = som_loss(norm_distance_matrix, sigma_local)
				if b==len(data_loader)-1 and self.wandb_log:
					pil_image = plotter.create_pil_image()
					wandb.log({	
						"weights": wandb.Image(pil_image),
						"loss" : loss.item()
					})

				loss = torch.mul(lr_local, loss)
				loss.backward()
				torch.nn.utils.clip_grad_value_(self.model.parameters(), 10) #TODO verify
				optimizer.step()
				optimizer.zero_grad()

		if wandb.run is not None:
			wandb.finish()
		return

