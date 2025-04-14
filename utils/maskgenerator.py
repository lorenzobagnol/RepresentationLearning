import torch
import torch.nn as nn


class MaskManager(nn.Module):
	"""
	Manages the creation and application of dropout masks.

	This module generates and stores a dropout mask based on a given
	probability.  The mask can be reused across multiple forward passes, or
	regenerated with a new probability.  It's designed to efficiently handle
	dropout within other neural network modules.
	"""

	def __init__(self, dropout_probability=0.0):
		"""
		Initializes the MaskManager.

		Args:
			- dropout_probability (float, optional): The initial dropout
			  probability.  Defaults to 0.0 (no dropout).
		"""
		super(MaskManager, self).__init__()
		self.dropout_probability = dropout_probability

	def forward(self, x):
		"""
		Generates (if needed) and returns a dropout mask.  The mask is applied
		element-wise to the input tensor to perform dropout.

		Args:
			- x (torch.Tensor): The input tensor.  The mask will have the same
			  shape as the *second* dimension of x, and a leading dimension of
			  1.  This is because the mask is applied to the *output* of a
			  linear layer.
		
		Returns:
			torch.Tensor: The dropout mask.
		"""
		p = self.dropout_probability

		
		self.dropout_probability = p
		# Create a Bernoulli distribution with probability (1-p) of keeping
		# a unit.  The mask has the same shape as the second dimension of x
		# (the number of neurons in the layer).  `x[0]` is used to get the
		# shape of the features, assuming x is of shape (batch_size,
		# feature_size)
		mask = torch.bernoulli(torch.empty_like(x[0]), 1 - p)
		# Reshape the mask to (1, feature_size) to be compatible for
		# broadcasting during multiplication
		#mask = self.mask.reshape(1, -1)

		return mask

	def update_dropout_probability(self, new_probability):
		"""
		Updates the dropout probability.

		Args:
			new_probability (float): The new dropout probability.
		"""
		self.dropout_probability = new_probability