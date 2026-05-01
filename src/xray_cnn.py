import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"
checkpoints_dir = "./checkpoints"

class XrayModel(nn.Module):
	def __init__(self, num_labels=14, xray_view_dim=5, weights=None):
		super().__init__()

		self.xray_view_dim = xray_view_dim

		self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
		old_weights = self.densenet.features.conv0.weight.data
		self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.densenet.features.conv0.weight.data = old_weights.mean(dim=1, keepdim=True)

		num_features = self.densenet.classifier.in_features
		self.densenet.classifier = nn.Identity()
		self.classifier = nn.Linear(num_features + xray_view_dim, num_labels)

		self.metanet = nn.Sequential(
			nn.Linear(xray_view_dim, 32),
			nn.ReLU()
		)

		self.classifier = nn.Sequential(
			nn.Linear(32 + num_features, 512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.25),

			nn.Linear(512, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(0.25),

			nn.Linear(256, num_labels)
		)

		if weights is not None:
			self.load_state_dict(weights)

	def forward(self, x: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
		if isinstance(x, tuple):
			x, xray_view = x
		else:
			xray_view = torch.zeros((x.shape[0], self.xray_view_dim), device=x.device)

		x = self.densenet(x)
		xray_view = self.metanet(xray_view.to(x.dtype))

		x = torch.cat([x, xray_view], dim=1)
		x = self.classifier(x)

		return x
