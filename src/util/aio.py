import asyncio
import logging

_tasks: set[asyncio.Task] = set()
_logger = logging.getLogger("sim.tasks")


def spawn(coro, logger: logging.Logger = None, name: str = None) -> asyncio.Task:
	"""create_task that keeps a strong reference and logs a traceback the moment the task
	fails. Bare fire-and-forget tasks report unretrieved exceptions only when the task
	object is garbage-collected — exception tracebacks form reference cycles, so a quiet
	hung process may never print them at all."""
	task = asyncio.create_task(coro, name=name)
	_tasks.add(task)

	def _on_done(t: asyncio.Task):
		_tasks.discard(t)
		if t.cancelled():
			return
		exc = t.exception()
		if exc is not None:
			(logger or _logger).error("Task %r died with an exception", t.get_name(), exc_info=exc)

	task.add_done_callback(_on_done)
	return task
