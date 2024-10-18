import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import math
import PIL
import matplotlib.pyplot as plt
from typing import Sequence, Union

from utils.plotter import Plotter
from models.som import SOM
from models.stm import STM



class SOMTrainer():
	
	def __init__(self, model: Union[SOM,STM], wandb_log: bool, clip_images: bool = False):
		self.model = model
		self.wandb_log = wandb_log
		self.clip_images = clip_images

	def available_training_modes(self):
		if isinstance(self.model, STM):
			return ["LifeLong", "pytorch_batch"]
		if isinstance(self.model, SOM):
			return ["simple_batch", "online", "pytorch_batch"]

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

		wandb.config.update(kwargs)
		wandb.config.update({"sigma": self.model.sigma})
		
		for it in range(kwargs["EPOCHS"]):
			for i, el in tqdm(enumerate(dataset_train), f"epoch {it+1}", len(dataset_train)):
				x=el[0]
				# look for the best matching unit (BMU)
				dists = torch.pairwise_distance(x, self.model.weights, p=2)
				_, bmu_index = torch.min(dists, 0)
				bmu_loc = self.model.locations[bmu_index,:]
				bmu_loc = bmu_loc.squeeze()
				
				learning_rate_op = np.exp(-it/kwargs["DECAY_RATE"])
				alpha_op = kwargs["ALPHA"] * learning_rate_op
				sigma_op = self.model.sigma * learning_rate_op

				# θ(u, v, s) is the neighborhood function which gives the distance between the BMU u and the generic neuron v in step s
				bmu_distance_squares = torch.sum(torch.pow(self.model.locations.float() - torch.stack([bmu_loc for i in range(self.model.m*self.model.n)]).float(), 2), 1) # dim = self.model_dim = m*n
				neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
				learning_rate_op = alpha_op * neighbourhood_func

				learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1]*np.ones(self.model.input_data.dim) for i in range(self.model.m*self.model.n)]) # dim = (m*n, input_dim)
				delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.model.m*self.model.n)]) - self.model.weights))       # element-wise multiplication -> dim = (m*n, input_dim)
				new_weights = torch.add(self.model.weights, delta)
				self.model.weights = torch.nn.Parameter(new_weights)
				if self.wandb_log:
					plotter = Plotter(self.model, self.clip_images)
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

		wandb.config.update(kwargs)
		wandb.config.update({"sigma": self.model.sigma})

		print("Creating a DataLoader object from dataset", end=" ",flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,)
		print("\u2713 \n",flush=True)

		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epoch"):
			for batch in data_loader:
				neighbourhood_func = self.neighbourhood_batch(batch[0],  kwargs["DECAY_RATE"], iter_no)
				# update weights
				new_weights = torch.matmul(neighbourhood_func.T, batch[0]) # (som_dim, batch_size)x(batch_size, input_dim) = (som_dim, input_dim)
				norm = torch.sum(neighbourhood_func, 0) # som_dim
				self.weights = torch.nn.Parameter(torch.div(new_weights.T, norm).T)
			if self.wandb_log:
				plotter = Plotter(self.model, self.clip_images)
				pil_image = plotter.create_pil_image()
				wandb.log({"weights": wandb.Image(pil_image)})
		if wandb.run is not None:
			wandb.finish()
		return

	def train_pytorch_batch(self, dataset_train: Dataset, dataset_val: Dataset, **kwargs):
		"""
		Train the STM using PyTorch's built-in optimizers and backpropagation.

		Args:
			dataset_train (Dataset): Dataset for training.
			dataset_val (Dataset): The validation dataset used to evaluate the model's competence after training.
			**kwargs: Keyword arguments for various training hyperparameters, including:
				
		Returns:
			None: This function does not return any values, but it updates the model's weights and logs progress.
		
		"""
		print("\nSTM training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		wandb.config.update(kwargs)
		wandb.config.update({"sigma": self.model.sigma})

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,)
		print("\u2713", flush=True)
		print("\n\n\n")
		self.model.train()
		optimizer = torch.optim.Adam(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epochs", leave=True, position=0):
			for b, batch in enumerate(data_loader):
				neighbourhood_func = self.model.neighbourhood_batch(batch[0], kwargs["DECAY_RATE"], iter_no)
				if isinstance(self.model, STM):
					target_dist = self.model.target_distance_batch(batch, kwargs["DECAY_RATE"], iter_no)
					weight_function = torch.mul(neighbourhood_func, target_dist)
				else:
					weight_function = neighbourhood_func
				distance_matrix = batch[0].unsqueeze(1).expand((batch[0].shape[0], self.model.weights.shape[0], batch[0].shape[1])) - self.model.weights.unsqueeze(0).expand((batch[0].shape[0], self.model.weights.shape[0], batch[0].shape[1])) # dim = (batch_size, som_dim, input_dim) 
				norm_distance_matrix = torch.sqrt(torch.sum(torch.pow(distance_matrix,2), 2)) # dim = (batch_size, som_dim) 
				loss = torch.mul(1/2,torch.sum(torch.mul(weight_function,norm_distance_matrix)))

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

			if self.wandb_log:
				plotter = Plotter(self.model, self.clip_images)
				pil_image = plotter.create_pil_image()
				wandb.log({	
					"weights": wandb.Image(pil_image),
					"loss" : loss.item()
				})
		if wandb.run is not None:
			wandb.finish()
		return

	def train_LifeLong(self, dataset_train: Dataset, dataset_val: Dataset, **kwargs):
		"""
		Train the STM using a Continual Learning approach. The dataset is divided basing on labels and the training is divided too. PyTorch's built-in optimizers and backpropagation.

		Args:
			dataset_train (Dataset): Dataset for training.
			dataset_val (Dataset): The validation dataset used to evaluate the model's competence after training.
			**kwargs: Keyword arguments for various training hyperparameters, including:
				
		Returns:
			None: This function does not return any values, but it updates the model's weights and logs progress.
		
		"""

		print("\nSTM LifeLong learning with batch mode and pytorch optimizations is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		wandb.config.update(kwargs)
		wandb.config.update({"sigma": self.model.sigma})

		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,)
		print("\u2713 \n", flush=True)
		print("\n\n\n")
		self.model.train()
		
		labels = list(set(dataset_train.targets.detach().tolist()))
		for i in range(len(labels)):
			if i not in labels:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		rep = math.ceil(len(labels)/kwargs["SUBSET_SIZE"])
		optimizer = torch.optim.Adam(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(kwargs["LR_GLOBAL_BASELINE"],math.exp(-kwargs["ALPHA"]*epoch)))
		for i in range(rep):
			print("Training on labels in range:\t"+str(i*kwargs["SUBSET_SIZE"]) +" <= label < "+str((i+1)*kwargs["SUBSET_SIZE"]))
			if kwargs["DISJOINT_TRAINING"]:
				indices = torch.where((dataset_train.targets>=i*kwargs["SUBSET_SIZE"]) & (dataset_train.targets<(i+1)*kwargs["SUBSET_SIZE"]))[0].tolist()
			else:
				indices = torch.where(dataset_train.targets<(i+1)*kwargs["SUBSET_SIZE"])[0].tolist()

			subset_lll=torch.utils.data.Subset(dataset_train, indices)
			print("This subset contains "+str(len(subset_lll))+" elements.")
			data_loader = torch.utils.data.DataLoader(subset_lll,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											)
			
			# lr_global = max(kwargs["LR_GLOBAL_BASELINE"],math.exp(-kwargs["ALPHA"]*i))
			sigma_global = max(self.model.sigma*math.exp(-kwargs["ALPHA"]*i),kwargs["SIGMA_BASELINE"])
			print("lr: "+str(optimizer.param_groups[0]['lr']))
			print("sigma: "+str(sigma_global))
			for iter_no in tqdm(range(kwargs["EPOCHS_PER_SUBSET"]), desc=f"Epochs", leave=True, position=0):
				lr_local = math.exp(-kwargs["BETA"]*iter_no) # *lr_global
				sigma_local = sigma_global*math.exp(-kwargs["BETA"]*iter_no)
				for b, batch in enumerate(data_loader):
					neighbourhood_func = self.model.neighbourhood_batch(batch[0], radius=sigma_local)
					target_dist = self.model.target_distance_batch(batch, radius=sigma_local)
					weight_function = torch.mul(neighbourhood_func, target_dist)
					distance_matrix = batch[0].unsqueeze(1).expand((batch[0].shape[0], self.model.weights.shape[0], batch[0].shape[1])) - self.model.weights.unsqueeze(0).expand((batch[0].shape[0], self.model.weights.shape[0], batch[0].shape[1])) # dim = (batch_size, som_dim, input_dim) 
					norm_distance_matrix = torch.sqrt(torch.sum(torch.pow(distance_matrix,2), 2)) # dim = (batch_size, som_dim) 
					loss = torch.mul(1/2*lr_local,torch.sum(torch.mul(weight_function, norm_distance_matrix)))

					loss.backward()
					optimizer.step()
					optimizer.zero_grad()
	
				if self.wandb_log:
					plotter = Plotter(self.model, self.clip_images)
					pil_image = plotter.create_pil_image()
					wandb.log({	
						"weights": wandb.Image(pil_image),
						"loss" : loss.item(),
						#"local competence" : local_competence
					})

			scheduler.step()
			
			checkpoint_path= os.path.join(os.path.curdir,"checkpoint.pt")
			torch.save({
				'label_range': i, 
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				}, checkpoint_path)
			
				# with torch.no_grad():
				# 	local_competence=self._compute_local_competence(val_set=dataset_val, label=i, batch_size=batch_size)
				# 	print(local_competence)
					
				# if local_competence < 0.0001:
				# 	print("Training interrupted for this subest after "+str(iter_no)+" epochs.")
				# 	print("Small local competence obtained : "+str(local_competence)+"\n")
				# 	stop_flag = True
				# 	break

		if wandb.run is not None:
			wandb.finish()
		return
