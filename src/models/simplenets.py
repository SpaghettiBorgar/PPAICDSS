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


class LeNet(nn.Module):
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
