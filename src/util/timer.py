import inspect
from time import time


class Timer:
	def __init__(self, print=True):
		self.start = None
		self.end = None
		self.print = print

	@property
	def elapsed(self):
		return self.end - self.start

	def __enter__(self):
		self.start = time()
		return self

	def __exit__(self, type=None, value=None, traceback=None):
		self.end = time()
		if self.print:
			frame = inspect.stack()[1]
			print(f"[{inspect.getmodule(frame).__name__}:{frame.function}] Elapsed: {self.elapsed:.2f}s")
