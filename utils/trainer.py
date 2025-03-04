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
	
	def __init__(self, model: Union[SOM,STM], device, wandb_log: bool, clip_images: bool = False):
		self.model = model
		self.wandb_log = wandb_log
		self.clip_images = clip_images
		self.device = device

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

		if self.wandb_log:
			wandb.config.update(kwargs)
		
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
				bmu_distance_squares = torch.sum(torch.pow(self.model.locations.float() - bmu_loc.unsqueeze(0).float(), 2), 1) # dim = self.model_dim = m*n
				neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
				learning_rate_op = alpha_op * neighbourhood_func

				learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1]*np.ones(self.model.input_data.dim) for i in range(self.model.m*self.model.n)]) # dim = (m*n, input_dim)
				delta = torch.mul(learning_rate_multiplier, x.unsqueeze(0) - self.model.weights)       # element-wise multiplication -> dim = (m*n, input_dim)
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

		if self.wandb_log:
			wandb.config.update(kwargs)

		print("Creating a DataLoader object from dataset", end=" ",flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,)
		print("\u2713 \n",flush=True)

		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epoch"):
			sigma_local = self.model.sigma*math.exp(-kwargs["BETA"]*iter_no)
			for batch in data_loader:
				neighbourhood_func = self.model.neighbourhood_batch(batch, sigma_local)
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

		if self.wandb_log:
			wandb.config.update(kwargs)


		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											drop_last=True)
		print("\u2713", flush=True)
		print("\n\n\n")
		self.model.train()
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epochs", leave=True, position=0):
			lr_local = math.exp(-kwargs["BETA"]*iter_no)
			sigma_local = self.model.sigma*math.exp(-kwargs["BETA"]*iter_no)
			for b, batch in enumerate(data_loader):
				inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
				norm_distance_matrix = self.model(inputs)
				if isinstance(self.model, STM):
					match kwargs["MODE"]:
						case "STC":
							weight_function = self.model.neighbourhood_batch_vieri(norm_distance_matrix, targets, radius=sigma_local)
						case "Base":
							neighbourhood_func = self.model.neighbourhood_batch(norm_distance_matrix, radius=sigma_local)
							target_dist = self.model.target_distance_batch(targets, radius=kwargs["target_radius"])
							weight_function = torch.mul(neighbourhood_func, target_dist)
						case "BGN": # not working. Learn where it shouldn't learn
							weight_function = self.model.target_and_bmu_weighted_batch(norm_distance_matrix, targets, radius=sigma_local)
						case "Base_Norm": # not working. Often divides by zero
							neighbourhood_func = self.model.neighbourhood_batch(norm_distance_matrix, radius=sigma_local)
							target_dist = self.model.target_distance_batch(targets, radius=kwargs["target_radius"])
							weight_function = torch.mul(neighbourhood_func, target_dist)
							max_weight_function = torch.max(weight_function,1).values # (batch_size, som_dim)
							if torch.max(max_weight_function)==0:
								print("loss normalization zero everywhere")
								if max(torch.sum(weight_function,1))==0:
									print("also weight is 0")
								continue
							weight_function = torch.div(weight_function, max_weight_function.unsqueeze(1))
						case "Base-STC":
							weight_function = self.model.hybrid_weight_function(norm_distance_matrix, targets, radius=sigma_local)
						case "STC-modified":
							weight_function = self.model.neighbourhood_batch_vieri_modified(norm_distance_matrix, targets, radius=sigma_local, target_radius=kwargs["target_radius"])
							
				else:
					weight_function = self.model.neighbourhood_batch(norm_distance_matrix, sigma_local)

				loss = torch.mul(1/2,torch.sum(torch.mul(weight_function, norm_distance_matrix)))

				if b==len(data_loader)-1 and self.wandb_log:
					plotter = Plotter(self.model, self.clip_images)
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
		return plotter

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

		if self.wandb_log:
			wandb.config.update(kwargs)

		self.model.train()
		
		labels = list(set(dataset_train.targets.detach().tolist()))
		for i in range(len(labels)):
			if i not in labels:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		
		rep = math.ceil(len(labels)/kwargs["SUBSET_SIZE"])
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
											drop_last=True
											)
			with torch.no_grad():
				initial_local_error = self.compute_local_competence(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
			for iter_no in tqdm(range(kwargs["EPOCHS_PER_SUBSET"]), desc=f"Epochs", leave=True, position=0):
				log_flag=iter_no==kwargs["EPOCHS_PER_SUBSET"]-1
				with torch.no_grad():
					actual_local_error = self.compute_local_competence(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
				lr_local = actual_local_error/initial_local_error
				sigma_local = max(self.model.sigma*actual_local_error/initial_local_error, 0.7)
				for b, batch in enumerate(data_loader):
					inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
					norm_distance_matrix = self.model(inputs)
					match kwargs["MODE"]:
						case "STC":
							weight_function = self.model.neighbourhood_batch_vieri(norm_distance_matrix, targets, radius=sigma_local)
						case "Base":
							neighbourhood_func = self.model.neighbourhood_batch(norm_distance_matrix, radius=sigma_local)
							target_dist = self.model.target_distance_batch(targets, radius=kwargs["target_radius"])
							weight_function = torch.mul(neighbourhood_func, target_dist)
						case "BGN": # not working. Learn where it shouldn't learn
							weight_function = self.model.target_and_bmu_weighted_batch(norm_distance_matrix, targets, radius=sigma_local)
						case "Base_Norm": # not working. Often divides by zero
							neighbourhood_func = self.model.neighbourhood_batch(norm_distance_matrix, radius=sigma_local)
							target_dist = self.model.target_distance_batch(targets, radius=kwargs["target_radius"])
							weight_function = torch.mul(neighbourhood_func, target_dist)
							max_weight_function = torch.max(weight_function,1).values # (batch_size, som_dim)
							if torch.min(max_weight_function)==0:
								print("loss normalization contains zeros.")
								break
							weight_function = torch.div(weight_function, max_weight_function.unsqueeze(1))
						case "Base-STC":
							weight_function = self.model.hybrid_weight_function(norm_distance_matrix, targets, radius=sigma_local)
						case "STC-modified":
							weight_function = self.model.neighbourhood_batch_vieri_modified(norm_distance_matrix, targets, radius=sigma_local, target_radius=kwargs["target_radius"])
							
					loss = torch.mul(1/2,torch.sum(torch.mul(weight_function, norm_distance_matrix)))

					if b==len(data_loader)-1 and self.wandb_log:
						if log_flag:

							with torch.no_grad():
								local_error=self.compute_local_competence(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
							plotter = Plotter(self.model, self.clip_images)
							pil_image = plotter.create_pil_image()
							wandb.log({	
								"weights": wandb.Image(pil_image),
								"loss" : loss.item(),
								"competence" : local_error.item(),
							})
						else:
							wandb.log({	
								"loss" : loss.item(),
							})

					loss = torch.mul(lr_local, loss)
					loss.backward()
					optimizer.step()
					optimizer.zero_grad()

			# # not found checkpoint folder in server
			# checkpoint_path = os.path.join(os.path.curdir,"checkpoint", "checkpoint.pt")
			# torch.save({
			# 	'label_range': i, 
			# 	'model_state_dict': self.model.state_dict(),
			# 	'optimizer_state_dict': optimizer.state_dict(),
			# 	}, checkpoint_path)
			
					
				# if local_competence < 0.0001:
				# 	print("Training interrupted for this subest after "+str(iter_no)+" epochs.")
				# 	print("Small local competence obtained : "+str(local_competence)+"\n")
				# 	stop_flag = True
				# 	break
		if self.wandb_log:
			with torch.no_grad():
				loss_nei, loss_tar, loss_base = self.compute_total_competence(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"])
			wandb.log({	
				"loss_neighbourhood": loss_nei.item(),
				"loss_target": loss_tar.item(),
				"loss_base": loss_base.item(),
			})
		else: 
			with torch.no_grad():
				loss_nei, loss_tar, loss_base = self.compute_total_competence(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"])			
			with open("results.txt", "a") as f:
				f.write(f"target_radius:{kwargs['target_radius']}, MODE:{kwargs['MODE']}, loss_neighbourhood:{loss_nei.item()}, loss_target:{loss_tar.item()}, loss_base:{loss_base.item()}\n")
				
		if wandb.run is not None:
			wandb.finish()
		return
	

	def compute_local_competence(self, val_set: Dataset, label: int, batch_size: int):

		indices = torch.where(val_set.targets==label)[0].tolist()
		subset_val=torch.utils.data.Subset(val_set, indices)
		data_loader = torch.utils.data.DataLoader(subset_val,
										batch_size=batch_size,
										shuffle=False,
										)
		total_distance=0
		for b, batch in enumerate(data_loader):
			inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
			norm_distance_matrix = self.model(inputs) # (batch_size, som_dim)

			# look for the best matching unit (BMU)
			_, bmu_indices = torch.min(norm_distance_matrix, 1) # som_dim
			total_distance+=torch.sum(_)
		
		total_distance /= len(subset_val)
		# total_distance= math.exp()

		return total_distance
	
	def compute_total_competence(self, val_set: Dataset, batch_size: int):

		data_loader = torch.utils.data.DataLoader(val_set,
										batch_size=batch_size,
										shuffle=False,
										)
		loss_nei = 0
		loss_tar = 0
		loss_base = 0
		min_radius=0.7
		for b, batch in enumerate(data_loader):
			inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
			norm_distance_matrix = self.model(inputs) # (batch_size, som_dim)

			neighbourhood_func = self.model.neighbourhood_batch(norm_distance_matrix, radius=min_radius)
			target_dist = self.model.target_distance_batch(targets, radius=min_radius)
			weight_function = torch.mul(neighbourhood_func, target_dist)
			loss_nei += torch.mul(1/2,torch.sum(torch.mul(neighbourhood_func, norm_distance_matrix)))
			loss_tar += torch.mul(1/2,torch.sum(torch.mul(target_dist, norm_distance_matrix)))
			loss_base += torch.mul(1/2,torch.sum(torch.mul(weight_function, norm_distance_matrix)))

		loss_nei = torch.div(loss_nei, len(val_set))
		loss_tar = torch.div(loss_tar, len(val_set))
		loss_base = torch.div(loss_base, len(val_set))
		# total_distance= math.exp()

		return loss_nei, loss_tar, loss_base
	


