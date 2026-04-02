import signal
import sys

import training
import xray_training

signal.signal(signal.SIGUSR1, lambda sig, frame: training.do_save())
signal.signal(signal.SIGTERM, lambda sig, frame: training.do_save() or sys.exit(0))

if __name__ == '__main__':
	print("Starting")
	try:
		xray_training.train_xray_model("testing", None, "cuda")
	except KeyboardInterrupt:
		print("Terminating")