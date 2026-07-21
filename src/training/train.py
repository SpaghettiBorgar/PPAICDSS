print("Initializing")
import argparse
import signal
import sys
import warnings

import torch

from training import trainer
from training.xray import xray_training
from util.utils import auto_type

warnings.filterwarnings("ignore", category=UserWarning, message=r"Secure RNG turned off.*")


def setup_signals():
	signal.signal(signal.SIGUSR1, lambda sig, frame: trainer.save_model())
	signal.signal(signal.SIGTERM, lambda sig, frame: trainer.save_model() and trainer.save_logs() or sys.exit(0))


def parse_args():
	parser = argparse.ArgumentParser(description="Train Xray Model")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
	parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint to load")
	parser.add_argument("--phase", "-p", type=str, default='testing', help="Training phase parameter preset")
	parser.add_argument("--name", type=str, default=trainer.save_name, help="Name to use in save path template")
	parser.add_argument("--save-path", "-o", type=str, default=trainer.save_model_path, help="Path template for checkpoint output")
	parser.add_argument("--logs-path", type=str, default=trainer.save_logs_path, help="Path template for logs output")
	parser.add_argument("-P", action='append', default=[], help="Override training parameters")
	parser.add_argument("--seed", type=int, default=0, help="Global random seed")
	return parser.parse_args()


def main():
	import os
	args = parse_args()

	print(f"cpu_count = {os.cpu_count()}, sched_affinity = {os.sched_getaffinity(0)}")
	for k, v in os.environ.items():
		if 'thread' in k.lower():
			print(f"{k}={v}")

	trainer.save_name = args.name
	trainer.save_model_path = args.save_path
	trainer.save_logs_path = args.logs_path

	params = {k: auto_type(v, k) for (k, v) in [p.partition('=')[::2] for p in args.P]}

	params = {k: auto_type(v) for (k, v) in [p.partition('=')[::2] for p in args.P]}

	assert (('target_epsilon' in params) == ('target_delta' in params) == ('grad_norm' in params)
	        ) or (('grad_norm' in params) == ('noise_mult' in params))

	if args.seed is not None:
		import random
		random.seed(args.seed)
		torch.manual_seed(args.seed)

	setup_signals()

	print("Starting")
	try:
		xray_training.train_xray_model(args.phase, args.checkpoint, args.device, **params)
	except KeyboardInterrupt:
		try:
			print("Terminating")
			if input("Save model? (y/N): ").lower() == 'y':
				trainer.save_model()
			if input("Save logs? (y/N): ").lower() == 'y':
				trainer.save_logs()
		except KeyboardInterrupt:
			pass


if __name__ == '__main__':
	main()
