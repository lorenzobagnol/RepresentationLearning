import numpy as np
from typing import Sequence, Union, Tuple, Literal
import torch

class InputData():

	def __init__(self, shape: Tuple[int, int], channels: int = None, channel_range: Literal['Unit', 'RGB'] = 'Unit') -> None:
		
		assert ((isinstance(shape, int) and shape==0) or (isinstance(shape, tuple) and len(shape)==2)), "input dim should be zero (for datapoints images) or a tuple (for images)"
		
		self.dim1, self.dim2 = shape
		self.dim = self.dim1*self.dim2*channels
		self.channels=channels
		self.channel_range=channel_range


	def transform_data(self, x: torch.Tensor) -> torch.FloatTensor:
		if self.channel_range=="Unit":
			x /= 255.0
		if self.dim==1 or self.dim==3:
			return x
		return x.reshape(self.dim)
	
	def inverse_transform_data(self, x: torch.Tensor) -> torch.FloatTensor:
		if self.channels==1:
			return x.reshape(self.dim1, self.dim2)
		if self.channels==3:
			return x.reshape(self.channels, self.dim1, self.dim2).permute(1, 2, 0)
	
	def transform_dataset(self, dataset: torch.Tensor) -> torch.Tensor:
		transformed_data=[]
		for data in dataset:
			transformed_data.append(self.transform_data(data))
		transformed_data=torch.stack(transformed_data, dim=0)
		return transformed_data