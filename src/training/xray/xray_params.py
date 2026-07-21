import os
from typing import TypeAlias, Union, override

import torch
from opacus.validators import ModuleValidator
from torchvision.transforms import InterpolationMode, v2

from models.xray_cnn import XrayModel, get_latest_checkpoint, NORM_MEAN, NORM_STD
from training.params import Params
from training.xray.xray_data import CLASS_POS_WEIGHTS
from util.dp_compat import fix_inplace, convert_bn_to_gn

PhaseType: TypeAlias = Union[str, int, None]

XRAY_DEFAULT_PARAMS = dict(
	batches=0,
	shuffle=True,
	freeze_backend=False,
	save=True,
	criterion=torch.nn.BCEWithLogitsLoss,
	optimizer=torch.optim.AdamW,
	normalize=True,
	resolution=512
)


class XrayParams(Params):
	freeze_backend: bool
	resolution: int
	phase: PhaseType
	normalize: bool
	_transform: v2.Transform | None
	_model: torch.nn.Module | None

	def __init__(self, phase: PhaseType = None, **kwargs):
		self.phase = phase
		self._model = None
		self._transform = None
		super().__init__(**(XRAY_DEFAULT_PARAMS | (PHASES[phase] if phase is not None else dict()) | kwargs))
		if self.checkpoint == "latest":
			self.checkpoint = get_latest_checkpoint()

	def get_transform(self):
		if self._transform is None:
			self._transform = v2.Compose([
				v2.Resize(size=None, max_size=self.resolution, interpolation=InterpolationMode.BICUBIC),
				v2.ToImage(),
				v2.ToDtype(torch.float32, scale=True),
				# v2.CenterCrop([params.resolution, params.resolution])
				*([v2.Normalize(mean=[NORM_MEAN], std=[NORM_STD])] if self.normalize else [])
			])
		return self._transform

	@override
	def get_model(self) -> XrayModel:
		if self._model is None:
			self._model = XrayModel().to(self.device)
			if os.environ.get("DP_FIX_INPLACE", "0") == "1":
				print("fixing inplace ops")
				self._model = fix_inplace(self._model)
			if os.environ.get("DP_VALIDATOR_FIX", "0") == "1":
				print("converting BatchNorm to GroupNorm(1) with statistics folding")
				self._model = convert_bn_to_gn(self._model, num_groups=1)
				errors = ModuleValidator.validate(self._model)
				if errors:
					print(f"remaining opacus incompatibilities, applying ModuleValidator.fix(): {errors}")
					self._model = ModuleValidator.fix(self._model)
			self._model = XrayModel.load_weights(self._model, self.get_weights())
			if self.freeze_backend:
				for module in self._model.backend.modules():
					# norm affines stay trainable: stands in for the running-stat
					# adaptation a frozen BatchNorm backbone gets for free
					if isinstance(module, (torch.nn.GroupNorm, torch.nn.modules.batchnorm._BatchNorm)):
						continue
					for param in module.parameters(recurse=False):
						param.requires_grad = False

		return self._model

	@override
	def get_criterion(self):
		if self._criterion is None:
			self._criterion = self.criterion(**dict(pos_weight=CLASS_POS_WEIGHTS.to(self.device)))
		return self._criterion

	@override
	def get_optimizer(self):
		if self._optimizer is None:
			self._optimizer = self.optimizer(
				self.get_model().parameters(),
				**dict(lr=self.lr, weight_decay=self.weight_decay),
				**(dict(momentum=self.sgd_momentum) if self.optimizer == torch.optim.SGD else dict())
			)
		return self._optimizer


PHASES = {
	"testing": dict(
		batch_size=8,
		batches=10,
		epochs=3,
		resolution=224,
		lr=1e-2,
		weight_decay=1e-3,
		freeze_backend=True,
		save=False
	),
	'1': dict(
		batch_size=512,
		epochs=8,
		resolution=384,
		lr=1e-3,
		weight_decay=1e-2,
		freeze_backend=True,
	),
	'2': dict(
		batch_size=160,
		epochs=12,
		resolution=384,
		lr=1e-4,
		weight_decay=1e-3,
		freeze_backend=False,
	),
	'3': dict(
		batch_size=64,
		epochs=16,
		resolution=512,
		lr=1e-4,
		weight_decay=1e-4,
		freeze_backend=False,
	)
}
