import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from models.som import SOM
from models.stm import STMLoss
from utils.config import TopologicalAEConfig
from utils.inputdata import InputData
from utils.maskgenerator import MaskManager

class TopologicalAE(nn.Module):
	"""
	Topological Autoencoder model for unsupervised representation learning.

	This class implements an autoencoder architecture where the encoder maps
	input data to a latent space, which is topologically constrained through a
	custom mapping, and the decoder reconstructs the input data from the latent
	space.
	"""

	def __init__(self, tae_config: TopologicalAEConfig):
		"""
		Initialize the Topological Autoencoder with specified latent dimension.

		Args:
			som_dim (int): The dimensionality of the SOM in the latent space.
		"""
		
		super(TopologicalAE, self).__init__()

		self.mask = MaskManager()
		# Encoder layers
		self.encoder_conv1 = nn.Conv2d(
			in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1
		)
		self.encoder_conv2 = nn.Conv2d(
			in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
		)
		self.encoder_fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=tae_config.SOM_CONFIG.INPUT_DATA.dim)
		
		self.topological_map = SOM(tae_config.SOM_CONFIG)
		
		# Decoder layers
		self.decoder_fc2 = nn.Linear(in_features=256, out_features=32 * 7 * 7)
		
		self.decoder_deconv1 = nn.ConvTranspose2d(
			in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1
		)
		self.decoder_deconv2 = nn.ConvTranspose2d(
			in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1
		)


	def encode(self, input_tensor):
		"""
		Encodes the input data to the latent space.

		Args:
			input_tensor (torch.Tensor): The input data.
		
		Returns:
			torch.Tensor: The latent space variables
		"""
		x = F.relu(self.encoder_conv1(input_tensor))
		x = self.mask(x)
		x = F.relu(self.encoder_conv2(x))
		x = self.mask(x)
		x = x.view(x.size(0), -1)
		x = F.relu(self.encoder_fc1(x))
		x = self.mask(x)
		topological_output = self.topological_map(x)
		return x, topological_output

	def decode(self, latent_variable):
		"""
		Decodes the latent variables back to original data space.

		Args:
			latent_variable (torch.Tensor): The latent space variables.

		Returns:
			torch.Tensor: Reconstructed data.
		"""
		z = F.relu(self.decoder_fc2(latent_variable))
		z = self.mask(z)
		z = z.view(z.size(0), 32, 7, 7)
		z = F.relu(self.decoder_deconv1(z))
		z = self.mask(z)
		z = torch.sigmoid(self.decoder_deconv2(z))

		return z

	def forward(self, input_tensor):
		"""
		Executes the forward pass for the autoencoder.

		Args:
			- input_tensor (torch.Tensor): The input data.

		Returns:
			- (torch.Tensor, torch.Tensor): The reconstructed output and the
			  normalized code.
		"""
		z, topological_output = self.encode(input_tensor)
		bmu, bmu_loc = self.topological_map.find_bmu(topological_output)
		reconstructed_output = self.decode(z)
		return reconstructed_output, topological_output

	def set_dropout_probability(self, p):
		"""
		Sets the dropout probability for the mask manager.
		"""

		self.mask.update_dropout_probability(p)


class TopologicalAELoss():
	

	def __init__(self, model: TopologicalAE, device, mode, target_points):

		self.model = model
		self.device = device
		self.stm_loss = STMLoss(self.model.topological_map, device, mode, target_points)


	def __call__(self, x:torch.Tensor, reconstructed:torch.Tensor, topological_output, sigma_local:float, target_radius:float, labels):

		MSE = nn.MSELoss(reduction="sum")
		reconstruction_loss = MSE(
			reconstructed, x.reshape(reconstructed.shape)
		)
		
		map_loss = self.stm_loss(topological_output, labels, sigma_local, target_radius)

		return reconstruction_loss, map_loss
