import operator
from typing import Callable, Any
from typing import Iterable, Union

import torch
from torch.types import Number

NamedTensors = Iterable[tuple[str, torch.Tensor]]
ParamSource = Union[torch.nn.Module, NamedTensors, 'Weights']
ParamSourceOrScalar = Union[ParamSource, Number]


class Weights:
	src: NamedTensors | torch.nn.Module

	def __init__(self, src: ParamSource):
		self.src = src

	@classmethod
	def _to_params(cls, src: ParamSource) -> NamedTensors:
		return src.named_parameters() if isinstance(src, torch.nn.Module) else src

	@torch.no_grad()
	def apply(self, x: ParamSourceOrScalar, fn: Callable[[torch.Tensor, torch.Tensor | Number], Any]):
		if isinstance(x, Number):
			for (n1, p1) in self:
				fn(p1, x)
		else:
			for (n1, p1), (n2, p2) in zip(self, Weights._to_params(x), strict=True):
				if n1 != n2:
					raise ValueError(f"Parameter mismatch: {n1} vs {n2}")
				fn(p1, p2)
		return self

	def assign(self, x: ParamSourceOrScalar):
		return self.apply(x, lambda p1, p2: p1.copy_(p2))

	def detach(self):
		return Weights([(n, t.detach()) for (n, t) in self])

	def add(self, x: ParamSourceOrScalar):
		return self.apply(x, lambda p1, p2: p1.add_(p2))

	def sub(self, x: ParamSourceOrScalar):
		return self.apply(x, lambda p1, p2: p1.sub_(p2))

	def mul(self, x: ParamSourceOrScalar):
		return self.apply(x, lambda p1, p2: p1.mul_(p2))

	def div(self, x: ParamSourceOrScalar):
		return self.apply(x, lambda p1, p2: p1.div_(p2))

	@torch.no_grad()
	def _binop(self, rhs: ParamSourceOrScalar, op):
		def gen():
			if isinstance(rhs, Number):
				for (n1, p1) in self:
					yield op(p1, rhs)
			else:
				for (n1, p1), (n2, p2) in zip(self, Weights._to_params(rhs), strict=True):
					if n1 != n2:
						raise ValueError(f"Parameter mismatch: {n1} vs {n2}")
					yield op(p1, p2)

		return Weights(gen())

	def __add__(self, rhs: ParamSourceOrScalar):
		return self._binop(rhs, operator.add)

	def __sub__(self, rhs: ParamSourceOrScalar):
		return self._binop(rhs, operator.sub)

	def __mul__(self, rhs: ParamSourceOrScalar):
		return self._binop(rhs, operator.mul)

	def __floordiv__(self, rhs: ParamSourceOrScalar):
		return self._binop(rhs, operator.floordiv)

	def __truediv__(self, rhs: ParamSourceOrScalar):
		return self._binop(rhs, operator.truediv)

	def __iter__(self) -> NamedTensors:
		return iter(Weights._to_params(self.src))
