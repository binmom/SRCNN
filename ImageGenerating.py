import torch
import torch.nn as nn
import torch.nn.functional as F
from data_load import testset
from torchvision.utils import save_image

### Draw Generating Model


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


name = 'Baby'
test_loader = testset(name)

model = torch.load("model.pth")
with torch.no_grad():
	for data in test_loader:
		images, labels = data
		draw = model(images)
		draw = (draw+1)/2.0
		save_image(draw, name+'.png')
