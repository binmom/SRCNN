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

def data_load():
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

	data_path = './datasets/Train_Blurred/'
	target_path = './datasets/Train_Cropped/'
	total_dataset = torchvision.datasets.ImageFolder(root = data_path, transform = transform)
	target_dataset = torchvision.datasets.ImageFolder(root = target_path, transform = transform)

	image = list(total_dataset)

	for i in range(len(total_dataset)):
		total = list(image[i])
		target = list(target_dataset[i])
		total[1] = target[0]
		image[i] = tuple(total)

	train_loader = DataLoader(image, batch_size = 1, num_workers = 0, shuffle = True)

	return train_loader

def testset(name):
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
	data_path = './'+ 'testsets' +'/' + name
	target_path = './'+ 'targets' +'/' + name
	test_dataset = torchvision.datasets.ImageFolder(root = data_path, transform = transform)
	target_dataset = torchvision.datasets.ImageFolder(root = target_path, transform = transform)

	image = list(test_dataset)
	for i in range(len(test_dataset)):
		total = list(image[i])
		target = list(target_dataset[i])
		total[1] = target[0]
		image[i] = tuple(total)
	test_loader = DataLoader(image, batch_size = 1, num_workers = 0, shuffle = False)
	print (name, 'testing')
	return test_loader

