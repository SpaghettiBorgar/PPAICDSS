print("Initializing")
import signal
import sys
import os

print(os.getenv("DISPLAY", default="DISPLAY unset"))

import matplotlib

def _configure_matplotlib_backend() -> None:
	print(f"Backend is {matplotlib.get_backend()}")
	# Respect explicit user override if set.
	if os.getenv("MPLBACKEND"):
		return

	# Headless-safe fallback.
	if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
		matplotlib.use("Agg", force=True)
		return

	# Try interactive backends, then safe fallback.
	for backend in ("QtAgg", "TkAgg", "Agg"):
		try:
			matplotlib.use(backend, force=True)
			return
		except Exception:
			pass


# _configure_matplotlib_backend()
print(f"Backend is now {matplotlib.get_backend()}")
import torch

import training
from training.xray import xray_training

signal.signal(signal.SIGUSR1, lambda sig, frame: training.do_save())
signal.signal(signal.SIGTERM, lambda sig, frame: training.do_save() or sys.exit(0))

if __name__ == '__main__':
	print("Starting")
	try:
		xray_training.train_xray_model('testing', None, f'cuda:{min(1, torch.cuda.device_count() - 1)}')
	except KeyboardInterrupt:
		print("Terminating")