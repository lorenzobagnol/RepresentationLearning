from typing import Sequence, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import types






class AE(nn.Module):
	"""
	base class for AutoEncoder
	"""
	def __init__(self, input_dim: Tuple[int, int]):
		super(AE, self).__init__()
		self.encoder = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2),
			torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=2)
			)
		self.conv_features = 25*6
		self.fully_connected = torch.nn.Linear(in_features=self.conv_features, out_features=10)
		self.decoder = torch.nn.Sequential(
			torch.nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2, output_padding=1),
			torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=4, stride=2)
			)
		
	def forward(self, input: torch.Tensor):
		x = self.conv1(input)
		x = self.conv2(x)
		x = x.reshape(self.conv_features)