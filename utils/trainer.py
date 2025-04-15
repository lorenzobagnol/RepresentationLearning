import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import math
import random

from utils.plotter import TopologicalAEPlotter
from models.topological_AE import TopologicalAE, TopologicalAELoss
from models.stm import TargetPoints


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
				loss = torch.add(reconstruction_loss, kwargs["DELTA"]*map_loss)


				if b==len(data_loader)-1 and self.wandb_log:
					topological_map_image =  plotter.create_pil_image(target_points)
					wandb.log({	
						"som_weights": wandb.Image(topological_map_image),
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
		self.model.set_dropout_probability(0.7)

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
				scaling_factor = torch.min(actual_local_error/initial_local_error, torch.tensor(1))
				lr_local = scaling_factor
				sigma_local = max(kwargs["SIGMA"]*scaling_factor, 0.7)
				for b, batch in enumerate(data_loader):
					optimizer.zero_grad()
					inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
					reconstructed, topological_output = self.model(inputs)
					reconstruction_loss, map_loss = loss_function(inputs, reconstructed, topological_output, sigma_local, kwargs["TARGET_RADIUS"], labels)
					loss = torch.add(reconstruction_loss, kwargs["DELTA"]*map_loss)

					if b==len(data_loader)-1 and self.wandb_log:
						if iter_no==kwargs["EPOCHS_PER_SUBSET"]-1:
							topological_map_image =  plotter.create_pil_image(target_points)
							wandb.log({	
								"som_weights": wandb.Image(topological_map_image),
								"reconstruction_loss" : reconstruction_loss.item(),
								"map_loss" : map_loss.item(),
								"loss" : loss.item()
							})
						else:
							wandb.log({
								"reconstruction_loss" : reconstruction_loss.item(),
								"map_loss" : map_loss.item(),
								"loss" : loss.item()
							})

					loss = torch.mul(lr_local, loss)
					loss.backward()
					optimizer.step()

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
			inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
			reconstructed, topological_output = self.model(inputs)

			# look for the best matching unit (BMU)
			bmu_distance_sq, bmu_indices = torch.min(topological_output, 1) # batch_size
			total_distance+=torch.sum(bmu_distance_sq)
		
		total_distance /= len(val_set)

		return total_distance
	
