import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union
import PIL
import io
from PIL.Image import Image

from models.som import SOM


class SOMPlotter():

	def __init__(self, model: SOM, clip_image: bool = False):
		self.model = model
		self.clip_image = clip_image

	def create_image_grid(self) -> np.ndarray:
		"""
		Create an image grid from the SOM weights.
		
		Args:
			som (SOM): The model of Self-Organizing Map.
		
		Returns:
			numpy array: heigh*width*channels array representing the image grid.
		"""
		weights = self.model.get_weights().cpu()
		image_grid=torch.cat([torch.cat([self.model.input_data.inverse_transform_data(weights[i+(j*self.model.n)]) for i in range(self.model.n)], 0) for j in range(self.model.m)], 1)
		if self.clip_image:
			return np.clip(image_grid, 0, 1)
		return np.array(image_grid)
	
	def resize_image(self, image_grid: np.ndarray) -> plt.Figure:
		target_width = 800  
		target_height = 800  
		dpi_value = min(300, max(72, target_width / image_grid.shape[1]))
		figsize_x = target_width / dpi_value
		figsize_y = target_height / dpi_value
		fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi_value)
		ax.imshow(image_grid)
		ax.axis("off")

		return fig
	
	def create_pil_image(self) -> Image:
		self.image_grid = self.create_image_grid()
		fig = self.resize_image(self.image_grid)
		 # Save the figure to a buffer
		buf = io.BytesIO()
		fig.savefig(buf, format='png', bbox_inches='tight')
		buf.seek(0)  # Rewind the buffer to the beginning
		
		# Create a PIL image from the buffer
		pil_image = PIL.Image.open(buf).copy()
		plt.close(fig)  # Close the figure to free memory
		buf.close()  # Close the buffer
		return pil_image
		