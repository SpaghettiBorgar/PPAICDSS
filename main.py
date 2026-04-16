print("Initializing")
import signal
import sys

import training
import xray_training

signal.signal(signal.SIGUSR1, lambda sig, frame: training.do_save())
signal.signal(signal.SIGTERM, lambda sig, frame: training.do_save() or sys.exit(0))

if __name__ == '__main__':
	print("Starting")
	try:
		xray_training.train_xray_model('testing', None, f'cuda:{min(1, torch.cuda.device_count() - 1)}')
	except KeyboardInterrupt:
		print("Terminating")