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
		x = F.relu(self.encoder_conv2(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.encoder_fc1(x))
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
		z = z.view(z.size(0), 32, 7, 7)
		z = F.relu(self.decoder_deconv1(z))
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
		# bmu, bmu_loc = self.topological_map.find_bmu(topological_output)
		reconstructed_output = self.decode(z)
		return reconstructed_output, topological_output


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

if __name__ == "__main__":

	# Define the training loop
	def training_epoch(model, train_loader, optimizer, epoch):
		model.train()
		loss_function = TopologicalAELoss(model, device, 10)
		train_loss = 0
		neighborhood_std = 10 * (0.99**epoch)
		lr = 1 * (0.99**epoch)
		beta = 0.0
		for batch_idx, (data, labels) in enumerate(train_loader):
			data = data.to(device)
			optimizer.zero_grad()
			reconstructed, topological_output = model(data)
			losses = loss_function(
				data,
				reconstructed,
				topological_output,
				neighborhood_std,
				2,
				labels,
				beta,
			)
			losses *= lr
			loss = losses.mean()
			loss.backward()
			train_loss += loss.item()
			optimizer.step()
			if batch_idx % 100 == 0:
				print(
					"Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
						epoch,
						batch_idx * len(data),
						len(train_loader.dataset),
						100.0 * batch_idx / len(train_loader),
						loss.item() / len(data),
					)
				)
		print(
			"====> Epoch: {} Average loss: {:.4f}".format(
				epoch, train_loss / len(train_loader.dataset)
			)
		)

	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load MNIST dataset

	mnist_data = datasets.MNIST(
		"data", train=True, download=True, transform=transforms.ToTensor()
	)
	mnist_data = Subset(mnist_data, range(10))
	train_loader = torch.utils.data.DataLoader(
		mnist_data, batch_size=2, shuffle=True
	)

	# Initialize AE model
	latent_dim = 10 * 10
	model = TopologicalAE(latent_dim).to(device)

	# Define optimizer
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	# Train the VAE
	num_epochs = 1000
	for epoch in range(num_epochs):
		training_epoch(model, train_loader, optimizer, epoch)


	data = np.random.rand(1000, 2)
	kmeans = KMeans(
		init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42
	)
	kmeans.fit(data)
	points = torch.tensor(10 * kmeans.cluster_centers_)


	plt.ion()

	for i, point in enumerate(points):
		z = model.topological_map.backward(points)
		generated = model.decode(z.unsqueeze(0))
		plt.subplot(1, 10, i + 1)
		plt.imshow(generated[0][0].detach().numpy())