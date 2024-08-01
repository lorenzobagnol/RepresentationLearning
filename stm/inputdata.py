import numpy as np
from typing import Sequence, Union, Tuple, Literal
import torch

class InputData():

	def __init__(self, shape: Union[int, Tuple[int, int]], channels: int = None, channel_range: Literal['Unit', 'RGB'] = 'Unit') -> None:
		
		assert (isinstance(shape, int) or (isinstance(shape, tuple) and len(shape)==2)), "input dim should be and integer or a tuple"
		
		if isinstance(shape, int):
			self.type="int"
			self.dim = shape
		else:
			self.type="tuple"
			self.dim1, self.dim2 = shape
			self.dim = self.dim1*self.dim2
		self.channels=channels
		self.channel_range=channel_range


	def transform_data(self, x: torch.Tensor) -> torch.FloatTensor:
		if self.type=="int":
			return x.to(torch.float32)

		return x.reshape(self.dim).to(torch.float32)
	
	def inverse_transform_data(self, x: torch.Tensor) -> torch.FloatTensor:
		return x.reshape(self.dim1, self.dim2).to(torch.float32)
	
	def transform_dataset(self, dataset: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
		
		transformed_data=[]
		for data in dataset:
			transformed_data.append(self.transform_data(data))
		return transformed_data