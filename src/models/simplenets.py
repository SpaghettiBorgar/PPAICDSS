import torch.nn.functional as F
from torch import nn as nn


class SimpleNet(nn.Module):
	def __init__(self, resolution=32, num_classes=14):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, 8, 5),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Flatten(),
			nn.Linear((resolution - 4) ** 2 * 2, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		# x = x.view(x.size(0), -1)
		return self.net(x)


class LeNet3(nn.Module):
	"""
	Reference model used in "Deep Leakage from Gradients" (Zhu et al., 2019)
	Adapted from https://github.com/Smuch12/DLG-demo
	"""

	def __init__(self, resolution=32, num_classes=100):
		super().__init__()
		act = nn.Sigmoid
		self.body = nn.Sequential(
			nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
			act(),
			nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
			act(),
			nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
			act(),
		)
		self.fc = nn.Sequential(nn.Linear((resolution // 4) ** 2 * 12, num_classes))

	def forward(self, x):
		out = self.body(x)
		out = out.view(out.size(0), -1)
		return self.fc(out)


class LeNet1(nn.Module):
	"""
	Monochrome version of the above
	"""

	def __init__(self, resolution=32, num_classes=100):
		super().__init__()
		act = nn.Sigmoid
		self.body = nn.Sequential(
			nn.Conv2d(1, 12, kernel_size=5, padding=5 // 2, stride=2),
			act(),
			nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
			act(),
			nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
			act(),
		)
		self.fc = nn.Sequential(nn.Linear((resolution // 4) ** 2 * 12, num_classes))

	def forward(self, x):
		out = self.body(x)
		out = out.view(out.size(0), -1)
		return self.fc(out)


class SampleConvNet(nn.Module):
	"""
	From opacus MNIST example
	https://github.com/meta-pytorch/opacus/blob/main/examples/mnist.py
	"""

	def __init__(self, resolution=32, num_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
		self.conv2 = nn.Conv2d(16, 32, 4, 2)
		self._flattened_n = ((((resolution - 2) // 2 - 1) - 4) // 2)
		self.fc1 = nn.Linear(32 * self._flattened_n ** 2, 22 + num_classes)
		self.fc2 = nn.Linear(22 + num_classes, num_classes)

	def forward(self, x):
		# x of shape [B, 1, 28, 28]
		x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
		x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
		x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
		x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
		x = x.view(-1, 32 * self._flattened_n ** 2)  # -> [B, 512]
		x = F.relu(self.fc1(x))  # -> [B, 32]
		x = self.fc2(x)  # -> [B, 10]
		return x

	def name(self):
		return "SampleConvNet"
