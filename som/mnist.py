from som import SOM
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
from inputdata import InputData
import importlib
import inputdata
from tqdm import tqdm


input_data=InputData((28,28))

# data in .data and labels in .targets
train_MNIST_dataset = torchvision.datasets.MNIST(
	root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
	train=True,
	download=True,
	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
)
MNIST_dataset = torchvision.datasets.MNIST(
	root="C:\\Users\\loren\\Documenti\\Lorenzo\\CNR\\RepresentationLearning",
	train=False,
	download=True,
	transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), input_data.transform_data]),
)

n_iter = 5
m=10
n=10
som = SOM(m, n, input_data=input_data, niter=n_iter)

for iter_no in range(n_iter):
	#Train with each vector one by one
	for i, el in tqdm(enumerate(train_MNIST_dataset), f"epoch {iter_no+1}", len(train_MNIST_dataset)):
		if i==100:
			break
		som(el[0], iter_no)

# rearrange weight in a matrix called image_grid
image_grid = [[] for i in range(m)]
weights = som.get_weights()
locations = som.get_locations()

image_grid=torch.cat([torch.cat([input_data.inverse_transform_data(weights[i]) for i in range(n)], 0) for j in range(m)], 1)



#Plot
plt.imshow(image_grid)
plt.title('Color SOM')
plt.show()