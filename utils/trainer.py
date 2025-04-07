import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import math
import random
import matplotlib.pyplot as plt
from typing import Literal, Sequence, Union

from utils.plotter import SOMPlotter, TopologicalAEPlotter
from models.som import SOM, SOMLoss
from models.topological_AE import TopologicalAE, TopologicalAELoss
from models.stm import TargetPoints, STMLoss



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
		Train the STM using PyTorch's built-in optimizers and backpropagation.

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



class STMTrainer():
	

	def __init__(self, model: SOM, device, wandb_log: bool, clip_images: bool = False):

		self.model = model
		self.wandb_log = wandb_log
		self.clip_images = clip_images
		self.device = device
	

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
			plotter = SOMPlotter(self.model, self.clip_images)


		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											drop_last=True)
		targets = list(set(dataset_train.targets.detach().tolist()))
		print("\u2713", flush=True)
		print("\n\n\n")
		self.model.train()
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		target_points = TargetPoints(len(targets), self.device, self.model.m, self.model.n)

		stm_loss = STMLoss(self.model, self.device, mode=kwargs["MODE"], target_points=target_points)

		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epochs", leave=True, position=0):
			lr_local = math.exp(-kwargs["BETA"]*iter_no)

			sigma_local = kwargs["SIGMA"]*math.exp(-kwargs["BETA"]*iter_no)
			for b, batch in enumerate(data_loader):
				inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
				norm_distance_matrix = self.model(inputs)
				loss = stm_loss(norm_distance_matrix, labels, sigma_local=sigma_local, target_radius=kwargs["TARGET_RADIUS"])

				if b==len(data_loader)-1 and self.wandb_log:
					pil_image =  plotter.create_pil_image(target_points)
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
			plotter = SOMPlotter(self.model, self.clip_images)

		self.model.train()
		
		targets = list(set(dataset_train.targets.detach().tolist()))
		for i in range(len(targets)):
			if i not in targets:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		
		rep = math.ceil(len(targets)/kwargs["SUBSET_SIZE"])
		list_labels = [i for i in range(rep)]
		random.seed(kwargs["SEED"])
		random.shuffle(list_labels)

		target_points = TargetPoints(rep, self.device, self.model.m, self.model.n)

		stm_loss = STMLoss(self.model, self.device, mode=kwargs["MODE"], target_points=target_points)

		for i in list_labels:
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
				initial_local_error = self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
			for iter_no in tqdm(range(kwargs["EPOCHS_PER_SUBSET"]), desc=f"Epochs", leave=True, position=0):
				with torch.no_grad():
					actual_local_error = self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
				lr_local = actual_local_error/initial_local_error
				sigma_local = max(kwargs["SIGMA"]*actual_local_error/initial_local_error, 0.7)
				for b, batch in enumerate(data_loader):
					inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
					norm_distance_matrix = self.model(inputs)
					loss = stm_loss(norm_distance_matrix, labels, sigma_local=sigma_local, target_radius=kwargs["TARGET_RADIUS"])
					
					if b==len(data_loader)-1 and self.wandb_log:
						if iter_no==kwargs["EPOCHS_PER_SUBSET"]-1:
							with torch.no_grad():
								local_error=self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
							pil_image = plotter.create_pil_image(target_points)
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

		if self.wandb_log:
			with torch.no_grad():
				bmu_target_distance = self.compute_BMU_target_distance(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"])
				loss_nei = self.compute_errors(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"])
			wandb.log({	
				"loss_neighbourhood": loss_nei.item(),
				"distance_BMU_target": bmu_target_distance.item(),
			})

		if wandb.run is not None:
			wandb.finish()
		return
	

	def compute_errors(self, val_set: Dataset, batch_size: int, label: int =None):

		if label is not None:
			indices = torch.where(val_set.targets==label)[0].tolist()
			val_set=torch.utils.data.Subset(val_set, indices)

		data_loader = torch.utils.data.DataLoader(val_set,
										batch_size=batch_size,
										shuffle=False,
										)
		total_distance=0
		for b, batch in enumerate(data_loader):
			inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
			norm_distance_matrix = self.model(inputs) # (batch_size, som_dim)

			# look for the best matching unit (BMU)
			bmu_distance_sq, bmu_indices = torch.min(norm_distance_matrix, 1) # batch_size
			total_distance+=torch.sum(bmu_distance_sq)
		
		total_distance /= len(val_set)

		return total_distance
	

	def compute_BMU_target_distance(self, val_set: Dataset, batch_size: int, target_points, label: int =None):

		if label is not None:
			indices = torch.where(val_set.targets==label)[0].tolist()
			val_set=torch.utils.data.Subset(val_set, indices)

		data_loader = torch.utils.data.DataLoader(val_set,
										batch_size=batch_size,
										shuffle=False,
										)
		
		total_distance=0
		for b, batch in enumerate(data_loader):
			inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
			norm_distance_matrix = self.model(inputs) # (batch_size, som_dim)

			bmu_distance_sq, bmu_indices = torch.min(norm_distance_matrix, 1) # batch_size
			bmu_loc = torch.stack([self.model.locations[bmu_index,:] for bmu_index in bmu_indices]) # (batch_size, 2)

			target_loc = torch.stack([target_points[int(targ)] for targ in targets]) # (batch_size, 2) 

			distance_bmu_target = torch.sqrt(torch.sum(torch.pow(target_loc-bmu_loc,2),1)) # batch_size
			total_distance+=torch.sum(distance_bmu_target)
		
		total_distance /= len(val_set)

		return total_distance
	






class TopologicalAETrainer():
	

	def __init__(self, model: TopologicalAE, device, wandb_log: bool, clip_images: bool = False):

		self.model = model
		self.wandb_log = wandb_log
		self.clip_images = clip_images
		self.device = device


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
		print("\nTopologicalAE training with batch mode and pytorch optimizations is starting with hyper-parameters:")
		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		if self.wandb_log:
			wandb.config.update(kwargs)
			plotter = TopologicalAEPlotter(self.model, self.clip_images)


		print("Creating a DataLoader object from dataset", end="     ", flush=True)
		data_loader = torch.utils.data.DataLoader(dataset_train,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											drop_last=True)
		targets = list(set(dataset_train.targets.detach().tolist()))
		print("\u2713", flush=True)
		print("\n\n\n")

		self.model.train()
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		target_points = TargetPoints(len(targets), self.device, self.model.topological_map.m, self.model.topological_map.n)
		loss_function = TopologicalAELoss(self.model, self.device, kwargs["MODE"], target_points)
		for iter_no in tqdm(range(kwargs["EPOCHS"]), desc=f"Epochs", leave=True, position=0):
			lr_local = math.exp(-kwargs["BETA"]*iter_no)
			sigma_local = kwargs["SIGMA"]*math.exp(-kwargs["BETA"]*iter_no)
			for b, batch in enumerate(data_loader):
				optimizer.zero_grad()
				inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
				reconstructed, topological_output = self.model(inputs)
				reconstruction_loss, map_loss = loss_function(inputs, reconstructed, topological_output, sigma_local, kwargs["TARGET_RADIUS"], labels)
				loss = torch.add(reconstruction_loss, 0.05*map_loss)


				if b==len(data_loader)-1 and self.wandb_log:
					topological_map_image, reconstructed_image =  plotter.create_pil_image(target_points)
					wandb.log({	
						"som_weights": wandb.Image(topological_map_image),
						"decoder_output": wandb.Image(reconstructed_image),
						"reconstruction_loss" : reconstruction_loss.item(),
						"map_loss" : map_loss.item(),
						"loss" : loss.item()
					})

				loss = torch.mul(lr_local, loss)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

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

		print("\nTopologicalAE LifeLong learning with batch mode and pytorch optimizations is starting with hyper-parameters:")

		for key, value in kwargs.items():
			print(f"\u2022 {key} = {value}")
		print("\n\n\n")

		if self.wandb_log:
			wandb.config.update(kwargs)
			plotter = TopologicalAEPlotter(self.model, self.clip_images)

		self.model.train()
		
		labels = list(set(dataset_train.targets.detach().tolist()))
		for i in range(len(labels)):
			if i not in labels:
				raise Exception("Dataset labels must be consecutive starting from zero.")
		
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		
		rep = math.ceil(len(labels)/kwargs["SUBSET_SIZE"])
		list_labels = [i for i in range(rep)]
		random.seed(kwargs["SEED"])
		random.shuffle(list_labels)

		target_points = TargetPoints(rep, self.device, self.model.topological_map.m, self.model.topological_map.n, seed=kwargs["SEED"])
		loss_function = TopologicalAELoss(self.model, self.device, kwargs["MODE"], target_points)

		for i in list_labels:
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
				initial_local_error = self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
			for iter_no in tqdm(range(kwargs["EPOCHS_PER_SUBSET"]), desc=f"Epochs", leave=True, position=0):
				with torch.no_grad():
					actual_local_error = self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
				lr_local = actual_local_error/initial_local_error
				sigma_local = max(kwargs["SIGMA"]*actual_local_error/initial_local_error, 0.7)
				for b, batch in enumerate(data_loader):
					optimizer.zero_grad()
					inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
					reconstructed, topological_output = self.model(inputs)
					loss = loss_function(inputs, reconstructed, topological_output, sigma_local, kwargs["TARGET_RADIUS"], labels)
						
					if b==len(data_loader)-1 and self.wandb_log:
						if iter_no==kwargs["EPOCHS_PER_SUBSET"]-1:
							topological_map_image, reconstructed_image =  plotter.create_pil_image(target_points)
							wandb.log({	
								"som_weights": wandb.Image(topological_map_image),
								"decoder_output": wandb.Image(reconstructed_image),
								"loss" : loss.item()
							})
						else:
							wandb.log({	
								"loss" : loss.item(),
							})

					loss = torch.mul(lr_local, loss)
					loss.backward()
					optimizer.step()

		if wandb.run is not None:
			wandb.finish()
		return
	