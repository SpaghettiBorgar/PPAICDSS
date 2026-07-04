import os
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from opacus.validators import ModuleValidator

from util.dp_compat import make_model_dp_compatible, convert_dp_state_dict

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"
checkpoints_dir = "./checkpoints/xray_resnet"


def get_latest_checkpoint(checkpoints_dir=checkpoints_dir, filter: Callable[[str], bool] = lambda f: not f.startswith("fl")) -> str:
	return os.path.join(
		checkpoints_dir,
		sorted(f for f in os.listdir(checkpoints_dir)
		       if f.endswith('.pt') and filter(f))[-1])


class XrayModel(nn.Module):
	def __init__(self, num_classes=14, xray_view_dim=5, backend=lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), **_):
		super().__init__()

		self.xray_view_dim = xray_view_dim

		self.backend = backend()
		old_weights = self.backend.conv1.weight.data
		self.backend.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.backend.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)

		num_features = self.backend.fc.in_features
		self.backend.fc = nn.Identity()
		self.classifier = nn.Linear(num_features + xray_view_dim, num_classes)

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

			nn.Linear(256, num_classes)
		)

		# Make compatible with opacus DP
		# self.backend = ModuleValidator.fix(self.backend)
		# self.classifier = ModuleValidator.fix(self.classifier)
		# self.make_private_compatible()


	# def make_private_compatible(self):
		# make_model_dp_compatible(self)

	@classmethod
	def load_weights(cls, module, weights):
		if weights is not None:
			try:
				module.load_state_dict(weights)
			except RuntimeError:
				print("Converting DP weights")
				module.load_state_dict(convert_dp_state_dict(weights))
		return module


	def forward(self, x: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
		# for tuple input, make sure to fix the data loader with util.utils.fix_collate() to preserve tuples in batches
		try:
			x, xray_view = x
		except Exception:
			xray_view = torch.zeros((x.shape[0], self.xray_view_dim), device=x.device)
		
		x = self.backend(x)
		xray_view = self.metanet(xray_view.to(x.dtype))

		x = torch.cat([x, xray_view], dim=1)
		x = self.classifier(x)

		return x
