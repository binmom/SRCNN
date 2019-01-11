import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as  transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
from data_load import data_load, testset
from math import log10

### loading data

train_loader = data_load()

### Model

class SRCNN(nn.Module):
	# Structure

	#Input -> Conv1 -> Relu -> Conv2 -> Relu -> Conv3 -> PSNR

	def __init__(self):
		super(SRCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
		self.conv2 = nn.Conv2d(64, 32, 1, 1, 0)
		self.conv3 = nn.Conv2d(32, 3, 5, 1, 2)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.relu(self.conv2(out))
		out = self.conv3(out) # Generating should not have Relu (Surely)
		return out

net = SRCNN()
criterion = nn.MSELoss()

### Optimizer

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.80)
# optimizer = optim.Adam(net.parameters())

### training

print('Training Start')

for epoch in range(300):
	for i, data in enumerate(train_loader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

print('Finish Training')
# torch.save(net,"model.pth")

### test_loadeing

name = 'Baby'
test_loader = testset(name)

## test

# model = torch.load("model.pth")
MSE = 0.0
with torch.no_grad():
	for data in test_loader:
		images, labels = data
		outputs = model(images)
		MSE = criterion(outputs, labels)
max_ = 1
psnr = 10 * log10(max_**2 / MSE)
print('PSNR = ',psnr)
