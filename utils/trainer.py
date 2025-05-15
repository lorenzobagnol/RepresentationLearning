import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Sequence, Union
import kmeans_pytorch

from utils.plotter import SOMPlotter
from models.som import SOM
from models.stm import STMEfficacyLoss, TargetPoints, STMLoss

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
					weights_grid = plotter.create_image_grid()
					wandb.log({	
						"weights": wandb.Image(plotter.create_pil_image(weights_grid, target_points)),
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
		list_labels = [i for i in range(len(targets))]
		
		optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs["LEARNING_RATE"])
		
		random.seed(kwargs["SEED"])
		random.shuffle(list_labels)

		target_points = TargetPoints(len(list_labels), self.device, self.model.m, self.model.n)

		# Define hyperparameters
		efficacy_radial_sigma = 10
		efficacy_decay = 0.005
		efficacy_saturation_factor = 2.5
		# stm_loss = STMLoss(self.model, self.device, mode=kwargs["MODE"], target_points=target_points)
		stm_loss = STMEfficacyLoss(
								model=self.model,
								device=self.device,
								mode="Base",
								target_points=target_points,
								efficacy_radial_sigma=efficacy_radial_sigma,
								efficacy_decay=efficacy_decay,
								efficacy_saturation_factor=efficacy_saturation_factor,
							)

		rep = math.ceil(len(targets)/kwargs["SUBSET_SIZE"])

		tasks = ["task"+str(i) for i in range(rep)]




		# Define hyperparameters
		learning_rate = 0.001
		batch_size = 64
		epochs = 100
		input_dim = 784
		latent_dim = 20 * 20
		anchor_sigma = 1.2
		neigh_sigma_max = 40
		neigh_sigma_base = 0.7
		lr_max = 2
		lr_base = 0.001
		efficacy_radial_sigma = 10
		efficacy_decay = 0.005
		efficacy_saturation_factor = 2.5

		accuracy = []
		for i in range(rep):
			print("Training on labels in:\t"+str(list_labels[i*kwargs["SUBSET_SIZE"]:(i+1)*kwargs["SUBSET_SIZE"]]))
			if kwargs["DISJOINT_TRAINING"]:
				indices = torch.where(torch.isin(dataset_train.targets, torch.tensor(list_labels[i*kwargs["SUBSET_SIZE"]:(i+1)*kwargs["SUBSET_SIZE"]])))[0].tolist()
			else:
				indices = torch.where(torch.isin(dataset_train.targets, torch.tensor(list_labels[:(i+1)*kwargs["SUBSET_SIZE"]])))[0].tolist()

			subset_lll=torch.utils.data.Subset(dataset_train, indices)
			print("This subset contains "+str(len(subset_lll))+" elements.")
			data_loader = torch.utils.data.DataLoader(subset_lll,
											batch_size=kwargs["BATCH_SIZE"],
											shuffle=True,
											drop_last=True
											)
			
			
			for iter_no in tqdm(range(kwargs["EPOCHS_PER_SUBSET"]), desc=f"Epochs", leave=True, position=0):
				scaling_factor = math.exp(-kwargs["BETA"]*iter_no) #max(1, actual_local_error/initial_local_error)
				sigma_local = max(kwargs["SIGMA"]*scaling_factor, 0.7)
				for b, batch in enumerate(data_loader):
					inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
					norm_distance_matrix = self.model(inputs)
					loss = stm_loss.loss(
												norm_distance_matrix,
												neighbourhood_radius_baseline=0.7,
												radius=20,
												modulation_baseline=lr_base,
												modulation_max=lr_max,
												labels=labels,
												target_radius=kwargs["TARGET_RADIUS"],
											)
					#loss = stm_loss(norm_distance_matrix, labels, sigma_local=sigma_local, target_radius=kwargs["TARGET_RADIUS"])
					
					if b==len(data_loader)-1 and self.wandb_log:
						if iter_no==kwargs["EPOCHS_PER_SUBSET"]-1:
							with torch.no_grad():
								local_error=self.compute_errors(val_set=dataset_val, label=i, batch_size=kwargs["BATCH_SIZE"])
							weights_grid = plotter.create_image_grid()
							wandb.log({	
								"weights": wandb.Image(plotter.create_pil_image(weights_grid, target_points)),
								"efficacies": wandb.Image(plotter.create_pil_image(stm_loss._efficacies.cpu().detach().numpy().reshape(self.model.m, self.model.n))),
								"loss" : loss.item(),
								"competence" : local_error.item(),
							})
						else:
							wandb.log({	
								"loss" : loss.item(),
							})

					loss.backward()
					optimizer.step()
					optimizer.zero_grad()
			with torch.no_grad():	
				anchor_groups = self.get_anchor_groups(target_points, n_cluster=len(list_labels[:(i+1)*kwargs["SUBSET_SIZE"]]))
				
				accuracy.append(self.compute_accuracy(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"], anchor_groups=anchor_groups, list_labels=list_labels[:(i+1)*kwargs["SUBSET_SIZE"]]))
			print("Accuracy on the validation set "+str(list_labels[:(i+1)*kwargs["SUBSET_SIZE"]])+" is: "+str(accuracy))
		
		df_accuracy = pd.DataFrame(columns=tasks+[str(kwargs["SEED"])])
		# save on a dataframe the accuracy
		df_accuracy.loc[len(df_accuracy)] = accuracy + [kwargs["SEED"]]

		# write the accuracy on a csv file adding a line to the file
		df_accuracy.to_csv("accuracy_results", mode='a', header=False, index=False)
		
		if self.wandb_log:
			with torch.no_grad():
				bmu_target_distance = self.compute_BMU_target_distance(val_set=dataset_val, batch_size=kwargs["BATCH_SIZE"], target_points=target_points)
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
	

	def compute_BMU_target_distance(self, val_set: Dataset, batch_size: int, target_points: TargetPoints, label: int =None):

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

			target_loc = torch.stack([target_points.get_point_from_label(targ).value for targ in targets]) # (batch_size, 2) 

			distance_bmu_target = torch.sqrt(torch.sum(torch.pow(target_loc-bmu_loc,2),1)) # batch_size
			total_distance+=torch.sum(distance_bmu_target)
		
		total_distance /= len(val_set)

		return total_distance
	
	def compute_accuracy(self, val_set: Dataset, batch_size:int, anchor_groups: torch.Tensor, list_labels: list =None):
		"""
		Compute the accuracy of the model on the validation set.

		Args:
			val_set (Dataset): The validation dataset.
			batch_size (int): The batch size for data loading.
			target_points: Target points for the model.
			label (list, optional): Specific labels to compute accuracy for. Defaults to None.

		Returns:
			float: The accuracy of the model on the validation set.
		"""
		if list_labels is not None:
			indices = torch.where(torch.isin(val_set.targets, torch.tensor(list_labels)))[0].tolist()
			
			val_set=torch.utils.data.Subset(val_set, indices)

		data_loader = torch.utils.data.DataLoader(val_set,
										batch_size=batch_size,
										shuffle=False,
										)
		
		correct_predictions = 0
		total_samples = 0
		for b, batch in enumerate(data_loader):
			inputs, targets = batch[0].to(self.device), batch[1].detach().cpu()
			norm_distance_matrix = self.model(inputs)
			bmu, bmu_loc = self.model.find_bmu(norm_distance_matrix) # batch_size
			bmu_loc_index = torch.tensor([self.model.locations.tolist().index(bmu.tolist()) for bmu in bmu_loc])
			predictions = anchor_groups[bmu_loc_index] # batch_size
			correct_predictions += torch.sum(predictions == targets).item()
			total_samples += len(targets)

		accuracy = correct_predictions / total_samples
		return accuracy


	def get_anchor_groups(self, target_points: TargetPoints, n_cluster: int):
			"""Assigns each weight vector to an anchor group using k-means.

			Args:
				anchors (torch.Tensor): Anchor points tensor.

			Returns:
				torch.Tensor: Anchor group assignments for each weight vector.
			"""

			cluster_ids, cluster_centers = kmeans_pytorch.kmeans(
				X=self.model.weights,
				num_clusters=n_cluster,
				distance="euclidean",
				device=self.model.device,
			)

			side_length = self.model.m * self.model.n
			side_indices = torch.arange(side_length)
			# Stack the indices and cluster IDs
			coordinate_cluster_ids = torch.cat([self.model.locations.cpu(), cluster_ids.unsqueeze(1)], dim=1)

			# Calculate the mean coordinate for each cluster
			cluster_means = torch.stack(
				[
					coordinate_cluster_ids[coordinate_cluster_ids[:, 2] == x]
					.float()
					.mean(0)
					for x in range(n_cluster)
				]
			)

			# Assign each cluster to the nearest anchor
			anchors = torch.stack([point.value for point in target_points.points])
			cluster_to_anchor = (
				torch.norm(
					anchors.cpu().reshape(-1, 1, 2)
					- cluster_means[:, :2].reshape(1, -1, 2),
					dim=-1,
				)
				.min(0)
				.indices
			)

			# Assign each weight vector to the anchor group of its cluster
			anchor_groups = cluster_to_anchor[coordinate_cluster_ids[:, 2]]

			return anchor_groups


