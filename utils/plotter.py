import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence, Union
import PIL

from models.som import SOM
from models.stm import STM


class Plotter():

	def __init__(self, model: Union[SOM,STM], clip_image: bool = False):
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
		weights = self.model.get_weights()
		image_grid=torch.cat([torch.cat([self.model.input_data.inverse_transform_data(weights[i+(j*self.model.n)]) for i in range(self.model.n)], 0) for j in range(self.model.m)], 1)
		if self.clip_image:
			return np.clip(image_grid, 0, 1)
		return np.array(image_grid)
	
	def resize_image(self, image_grid: np.ndarray):
		target_width = 800  
		target_height = 800  
		dpi_value = min(300, max(72, target_width / image_grid.shape[1]))
		figsize_x = target_width / dpi_value
		figsize_y = target_height / dpi_value
		fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi_value)
		if isinstance(self.model, STM):
			base_font_size = 24  
			font_size = base_font_size * (dpi_value / 100)  
			for key, value in self.model.target_points.items():
				ax.text(value[0]*self.model.input_data.dim1, value[1]*self.model.input_data.dim2, str(key), ha='center', va='center',
					bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=font_size)
		ax.imshow(image_grid)
		ax.axis("off")

		return fig
	
	def create_pil_image(self):
		image_grid = self.create_image_grid()
		fig = self.resize_image(image_grid)
		fig.canvas.draw()
		pil_image=PIL.Image.frombytes('RGB', 
			fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
		plt.close(fig)
		return pil_image
	

