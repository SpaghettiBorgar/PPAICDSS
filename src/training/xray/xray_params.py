from typing import TypeAlias, Union, override

import torch
from torchvision.transforms import InterpolationMode, v2

import training.xray.xray_data as xray_data
from training.params import Params
from models.xray_cnn import XrayModel

PhaseType: TypeAlias = Union[str, int, None]

CLASS_WEIGHTS = xray_data.TOTAL_SAMPLES / torch.tensor(list(xray_data.CLASS_WEIGHTS.values())) - 1

XRAY_DEFAULT_PARAMS = dict(
	batches=0,
	shuffle=True,
	freeze_backend=False,
	save=True,
	criterion=torch.nn.BCEWithLogitsLoss,
	optimizer=torch.optim.AdamW
)


class XrayParams(Params):
	freeze_backend: bool
	resolution: int
	phase: PhaseType
	_transform: v2.Transform | None
	_model: torch.nn.Module | None

	def __init__(self, phase: PhaseType = None, **kwargs):
		super().__init__(**(XRAY_DEFAULT_PARAMS | (PHASES[phase] if phase is not None else dict()) | kwargs))
		self.phase = phase
		self._model = None
		self._transform = None

	def get_transform(self):
		if self._transform is None:
			self._transform = v2.Compose([
				v2.Resize(size=None, max_size=self.resolution, interpolation=InterpolationMode.BICUBIC),
				v2.ToImage(),
				v2.ToDtype(torch.float32, scale=True),
				# v2.CenterCrop([params.resolution, params.resolution])
			])
		return self._transform

	@override
	def get_model(self):
		if self._model is None:
			self._model = XrayModel(weights=self.get_weights()).to(self.device)
			if self.freeze_backend:
				for param in self._model.backend.parameters():
					param.requires_grad = False

		return self._model

	@override
	def get_criterion(self):
		if self._criterion is None:
			self._criterion = self.criterion(**dict(pos_weight=CLASS_WEIGHTS.to(self.device)))
		return self._criterion

	@override
	def get_optimizer(self):
		if self._optimizer is None:
			self._optimizer = self.optimizer(self.get_model().parameters(), **dict(lr=self.lr, weight_decay=self.weight_decay))
		return self._optimizer


PHASES = {
	"testing": dict(
		batch_size=4,
		batches=10,
		epochs=3,
		resolution=224,
		lr=1e-3,
		weight_decay=1e-3,
		freeze_backend=True,
		save=False
	),
	1: dict(
		batch_size=512,
		epochs=6,
		resolution=384,
		lr=1e-3,
		weight_decay=1e-3,
		freeze_backend=True,
	),
	2: dict(
		batch_size=160,
		epochs=10,
		resolution=384,
		lr=1e-4,
		weight_decay=1e-3,
		freeze_backend=False,
	),
	3: dict(
		batch_size=64,
		epochs=4,
		resolution=600,
		lr=1e-4,
		weight_decay=1e-4,
		freeze_backend=False,
	)
}
