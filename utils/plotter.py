import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Union
import PIL
import io
from PIL.Image import Image

from models.stm import TargetPoints
from models.topological_AE import TopologicalAE 


class TopologicalAEPlotter():

	def __init__(self, model: TopologicalAE, clip_image: bool = False):
		self.model = model
		self.clip_image = clip_image

	
	def create_pil_image(self, target_points: TargetPoints=None) -> Tuple[Image, Image]:
		"""
		Create a PIL image of the topological map and of the reconstructed images of the AE model from the target points.
		
		Args:
			model (TopologicalAE): The model of Topological Autoencoder.
		
		Returns:
			PIL image: The PIL image of the topological map.
			PIL image: The PIL image of the generated images.
		"""
		topological_map_image = self.create_som_pil_image(target_points)
		return topological_map_image
	

	def create_image_grid(self) -> np.ndarray:
		"""
		Create an image grid from the decoding of SOM weights.
		
		Args:
			model (TopologicalAE): The model of Topological Autoencoder.
		
		Returns:
			numpy array: heigh*width*channels array representing the image grid.
		"""
		weights = self.model.topological_map.get_weights().detach()
		image_grid=torch.cat([torch.cat([self.model.decode(weights[i+(j*self.model.topological_map.n)].unsqueeze(0))[0,0] for i in range(self.model.topological_map.n)], 0) for j in range(self.model.topological_map.m)], 1).detach().cpu()
		if self.clip_image:
			return np.clip(image_grid, 0, 1)
		return np.array(image_grid)
	
	def resize_image(self, image_grid: np.ndarray, target_points: TargetPoints=None) -> plt.Figure:
		target_width = 800  
		target_height = 800  
		dpi_value = min(300, max(72, target_width / image_grid.shape[1]))
		figsize_x = target_width / dpi_value
		figsize_y = target_height / dpi_value
		fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi_value)
		output_image_dim = image_grid.shape 
		output_single_image_dim = (output_image_dim[0]//self.model.topological_map.n, output_image_dim[1]//self.model.topological_map.m)
		if target_points is not None:
			base_font_size = 24  
			font_size = base_font_size * (dpi_value / 100)  
			for point in target_points.points:
				ax.text(point.value.cpu()[0]*output_single_image_dim[0], point.value.cpu()[1]*output_single_image_dim[1], str(point.label), ha='center', va='center',
					bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=0),  fontsize=font_size)
		ax.imshow(image_grid)
		ax.axis("off")

		return fig
	
	def create_som_pil_image(self, target_points: TargetPoints=None) -> Image:
		self.image_grid = self.create_image_grid()
		fig = self.resize_image(self.image_grid, target_points)
		 # Save the figure to a buffer
		buf = io.BytesIO()
		fig.savefig(buf, format='png', bbox_inches='tight')
		buf.seek(0)  # Rewind the buffer to the beginning
		
		# Create a PIL image from the buffer
		pil_image = PIL.Image.open(buf).copy()
		plt.close(fig)  # Close the figure to free memory
		buf.close()  # Close the buffer
		return pil_image
		

