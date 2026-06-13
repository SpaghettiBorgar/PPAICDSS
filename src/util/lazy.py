import functools
import operator
from typing import Callable

_UNSET = object()


def make_unary(op):
	def method(self):
		return op(self._get())

	return method


def make_binary(op):
	def method(self, other):
		other = other._get() if isinstance(other, Lazy) else other
		return op(self._get(), other)

	return method


def make_rbinary(op):
	def method(self, other):
		other = other._get() if isinstance(other, Lazy) else other
		return op(other, self._get())

	return method


def make_itemgetter():
	def method(self, key):
		return self._get()[key]

	return method


def make_len():
	def method(self):
		return len(self._get())

	return method


def make_iter():
	def method(self):
		return iter(self._get())

	return method


def make_contains():
	def method(self, item):
		item = item._get() if isinstance(item, Lazy) else item
		return item in self._get()

	return method


def make_call():
	def method(self, *args, **kwargs):
		return self._get()(*args, **kwargs)

	return method


class LazyMeta(type):
	def __new__(mcls, name, bases, namespace):
		cls = super().__new__(mcls, name, bases, namespace)

		unary_ops = {
			"__neg__": operator.neg,
			"__pos__": operator.pos,
			"__abs__": operator.abs,
			"__invert__": operator.invert,
		}

		binary_ops = {
			"__add__": operator.add,
			"__sub__": operator.sub,
			"__mul__": operator.mul,
			"__truediv__": operator.truediv,
			"__floordiv__": operator.floordiv,
			"__mod__": operator.mod,
			"__pow__": operator.pow,
			"__and__": operator.and_,
			"__or__": operator.or_,
			"__xor__": operator.xor,
			"__lshift__": operator.lshift,
			"__rshift__": operator.rshift,
		}

		reverse_ops = {
			"__radd__": operator.add,
			"__rsub__": operator.sub,
			"__rmul__": operator.mul,
			"__rtruediv__": operator.truediv,
			"__rfloordiv__": operator.floordiv,
			"__rmod__": operator.mod,
			"__rpow__": operator.pow,
			"__rand__": operator.and_,
			"__ror__": operator.or_,
			"__rxor__": operator.xor,
			"__rlshift__": operator.lshift,
			"__rrshift__": operator.rshift,
		}

		comparisons = {
			"__eq__": operator.eq,
			"__ne__": operator.ne,
			"__lt__": operator.lt,
			"__le__": operator.le,
			"__gt__": operator.gt,
			"__ge__": operator.ge,
		}

		special = {
			**{name: make_unary(op) for name, op in unary_ops.items()},
			**{name: make_binary(op) for name, op in binary_ops.items()},
			**{name: make_rbinary(op) for name, op in reverse_ops.items()},
			**{name: make_binary(op) for name, op in comparisons.items()},
			"__getitem__": make_itemgetter(),
			"__len__": make_len(),
			"__iter__": make_iter(),
			"__contains__": make_contains(),
			"__call__": make_call(),
		}

		for name, fn in special.items():
			if name not in cls.__dict__:
				setattr(cls, name, fn)

		return cls


class Lazy(metaclass=LazyMeta):
	def __init__(self, fn: Callable, *args, **kwargs):
		self._value = _UNSET
		self._fn = functools.partial(fn, *args, **kwargs)

	def _get(self):
		if self._value is _UNSET:
			self._value = self._fn()
		return self._value

	def __getattr__(self, name):
		return getattr(self._get(), name)

	def __repr__(self):
		if self._value is _UNSET:
			return f"Lazy(<unevaluated {self._fn!r}>)"
		return f"Lazy({self._value!r})"
