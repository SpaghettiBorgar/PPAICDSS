import os
from typing import TypeAlias, Union, Type, Any

import torch

DeviceLikeType: TypeAlias = Union[str, torch.device, int]
CheckpointType: TypeAlias = Union[str, None]
SamplerType: TypeAlias = Union[torch.utils.data.Sampler, str]

CHECKPOINTS_DIR = os.getenv("CHECKPOINTS_DIR", default="./checkpoints")

DEFAULT_PARAMS = dict(
	device="cuda",
	sampler="default",
	save=False,
	batch_size=8,
	batches=0,
	epochs=3,
	checkpoint=None,
	lr=None,
	weight_decay=None
)


class Params:
	device: DeviceLikeType
	sampler: SamplerType
	save: bool
	batch_size: int
	batches: int
	epochs: int
	checkpoint: CheckpointType
	_weights: Any | None
	criterion: Type[torch.nn.Module]
	_criterion: torch.nn.Module | None
	lr: float
	weight_decay: float
	optimizer: Type[torch.optim.Optimizer]
	_optimizer: torch.optim.Optimizer | None

	def __init__(self, **kwargs):
		self.__dict__.update(DEFAULT_PARAMS | kwargs)
		self._weights = None
		self._criterion = None
		self._optimizer = None

	def _update(self):
		pass

	def get_model(self) -> torch.nn.Module:
		pass

	def get_weights(self):
		if self._weights is None:
			self._weights = None if self.checkpoint is None else torch.load(self.checkpoint, weights_only=True, map_location=self.device)
		return self._weights

	def get_criterion(self):
		if self._criterion is None:
			self._criterion = self.criterion()
		return self._criterion

	def get_optimizer(self):
		if self._optimizer is None:
			self._optimizer = self.optimizer(self.get_model().parameters(), **dict(lr=self.lr, weight_decay=self.weight_decay))

	def __or__(self, rhs):
		return Params(**(self.__dict__ | rhs.__dict__))

	def __ior__(self, rhs):
		self.__dict__ |= rhs.__dict__
		self._update()

	def __repr__(self):
		return str(self.__dict__)
