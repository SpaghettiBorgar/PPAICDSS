import operator
from typing import Callable, Any, Iterator, List
from typing import Iterable, Union

import numpy as np
import torch
from torch.types import Number

NamedTensors = Iterable[tuple[str, torch.Tensor]]
ParamSource = Union[torch.nn.Module, NamedTensors, 'Weights']
ParamSourceOrScalar = Union[ParamSource, Number]


class Weights(Iterable):
	src: NamedTensors | torch.nn.Module

	def __init__(self, src: ParamSource):
		self.src = src

	@classmethod
	def _to_params(cls, src: ParamSource) -> NamedTensors:
		return src.named_parameters() if isinstance(src, torch.nn.Module) else src

	def shapes(self) -> List[tuple[str, tuple[int, ...]]]:
		return [(n, tuple(p.shape)) for (n, p) in self]

	def flatten(self) -> tuple[np.ndarray, List[tuple[str, tuple[int, ...]]]]:
		ps = list(self)
		shapes = [(n, tuple(p.shape)) for (n, p) in ps]
		if not ps:
			return np.empty(0), shapes
		dev, dt = ps[0][1].device, ps[0][1].dtype
		if any(p.device != dev or p.dtype != dt for _, p in ps):
			raise ValueError("All parameters must share device and dtype to flatten")
		return np.concatenate([p.detach().cpu().numpy().reshape(-1) for _, p in ps]), shapes

	@classmethod
	def unflatten(cls, vec: np.ndarray, shapes: List[tuple[str, tuple[int, ...]]]) -> 'Weights':
		vec, off, ps = vec.reshape(-1), 0, []
		for n, s in shapes:
			k = 1
			for d in s:
				k *= d
			ps.append((n, torch.from_numpy(np.reshape(vec[off:off + k], s))))
			off += k
		if off != vec.size:
			raise ValueError(f"Vector has wrong size: {vec.size} vs expected {off}")
		return cls(ps)

	@torch.no_grad()
	def apply(self, x: ParamSourceOrScalar, fn: Callable[[torch.Tensor, torch.Tensor | Number], Any]):
		if isinstance(x, np.ndarray):
			x = Weights.unflatten(x, self.shapes())
		if isinstance(x, Number):
			for (n1, p1) in self:
				fn(p1, x)
		else:
			for (n1, p1), (n2, p2) in zip(self, Weights._to_params(x), strict=True):
				if n1 != n2:
					raise ValueError(f"Parameter mismatch: {n1} vs {n2}")
				fn(p1, p2.to(device=p1.device))
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
		if isinstance(rhs, torch.Tensor):
			rhs = Weights.unflatten(rhs, self.shapes())

		def gen():
			if isinstance(rhs, Number):
				for (n1, p1) in self:
					yield n1, op(p1, rhs)
			else:
				for (n1, p1), (n2, p2) in zip(self, Weights._to_params(rhs), strict=True):
					if n1 != n2:
						raise ValueError(f"Parameter mismatch: {n1} vs {n2}")
					yield n1, op(p1, p2.to(device=p1.device))

		return Weights(list(gen()))

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

	def __eq__(self, rhs):
		for (n1, p1), (n2, p2) in zip(self, Weights._to_params(rhs), strict=True):
			if n1 != n2:
				raise ValueError(f"Parameter mismatch: {n1} vs {n2}")
			if not p1.equal(p2):
				return False
		return True

	def __ne__(self, rhs):
		return not self.__eq__(rhs)

	def __iter__(self) -> Iterator[NamedTensors]:
		return iter(Weights._to_params(self.src))
