print("Initializing")
import argparse
import signal
import sys

import torch

from training import training
from training.xray import xray_training
from util.utils import auto_type


def setup_signals():
	signal.signal(signal.SIGUSR1, lambda sig, frame: training.do_save())
	signal.signal(signal.SIGTERM, lambda sig, frame: training.do_save() or sys.exit(0))


def parse_args():
	parser = argparse.ArgumentParser(description="Train Xray Model")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
	parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint to load")
	parser.add_argument("--phase", "-p", type=str, default='testing', help="Training phase parameter preset")
	parser.add_argument("--save-path", "-o", type=str, default=training.save_model_path, help="Path template for checkpoint output")
	parser.add_argument("--logs-path", type=str, default=training.save_logs_path, help="Path template for logs output")
	parser.add_argument("-P", action='append', default=[], help="Override training parameters")
	parser.add_argument("--seed", type=int, default=0, help="Global random seed")
	return parser.parse_args()


def main():
	args = parse_args()
	training.save_model_path = args.save_path
	training.save_logs_path = args.logs_path


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
		print("Terminating")


if __name__ == '__main__':
	main()
