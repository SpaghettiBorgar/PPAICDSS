import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def vis_batch(t, nrow: int = None, ax=None):
	from torchvision.utils import make_grid
	if nrow is None:
		nrow = int(math.ceil(math.sqrt(t.shape[0])))
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	img = make_grid(t, nrow=nrow).permute(1, 2, 0)
	im = ax.imshow(img)
	fig.colorbar(im, ax=ax)

	return fig, ax


def smooth_gaussian(values, sigma: float) -> np.ndarray:
	if sigma <= 0:
		return values
	window_size = int(10 * sigma + 1)
	if len(values) < window_size:
		return values
	x = np.arange(window_size) - window_size // 2
	kernel = np.exp(-0.5 * (x / sigma) ** 2)
	kernel /= kernel.sum()
	return np.convolve(values, kernel, mode="valid")


def flatten(l):
	for el in l:
		if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
			yield from flatten(el)
		else:
			yield el


def is_federated(values):
	try:
		return isinstance(values, list) and len(values) > 0 and isinstance(values[0], list) and len(values[0]) > 0 and isinstance(values[0][0], list)
	except (IndexError, TypeError):
		return False


def plot_federated(rounds, smoothing=0.0, ax=None, xlabel="batch", ylabel="loss", title=None, label=None):
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	current_x = 0
	labeled_participants = set()
	for r_idx, participants in enumerate(rounds):
		ax.axvline(x=current_x, color='gray', linestyle='--', alpha=0.5)
		ax.text(current_x, 1.0, f" R{r_idx}", transform=ax.get_xaxis_transform(), rotation=90, verticalalignment='top', fontsize='small')

		max_round_len = 0
		for p_idx, participant_data in enumerate(participants):
			flat_values = list(flatten(participant_data))
			if smoothing > 0:
				flat_values = smooth_gaussian(flat_values, smoothing)

			x_values = np.arange(len(flat_values)) + current_x
			p_label = None
			if p_idx not in labeled_participants:
				p_label = f"P{p_idx}"
				labeled_participants.add(p_idx)

			ax.plot(x_values, flat_values, label=p_label, color=f"C{p_idx}")
			max_round_len = max(max_round_len, len(flat_values))

		current_x += max_round_len

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	ax.legend()

	return fig, ax


def get_logs(path):
	with open(path, 'r') as f:
		return json.load(f)


def latest_logs_path(logs_dir: str | Path = "./logs"):
	logs_dir = Path(logs_dir)
	files = [p for p in logs_dir.iterdir() if p.is_file()]
	return get_logs(max(files, key=lambda p: p.stat().st_mtime))


def plot_logs(values: Iterable[Any], smoothing: float = 0., ax=None, xlabel="batch", ylabel="loss", title: str = None, label: str = None):
	if is_federated(values):
		return plot_federated(values, smoothing, ax, xlabel, ylabel, title, label)

	values = list(flatten(values))
	values = smooth_gaussian(values, smoothing)

	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	ax.plot(values, label=label)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title)
	if label:
		ax.legend()

	return fig, ax


def concat_logs(*logs: dict[str, Any], key: str = 'loss_history') -> list[float]:
	from functools import reduce
	return reduce(lambda x, y: x + y, (log.get(key, []) for log in logs), [])


def parse_args():
	parser = argparse.ArgumentParser(description="Visualize training logs")
	parser.add_argument("paths", nargs="*", help="Paths to log files or latest if omitted")
	parser.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing sigma")
	parser.add_argument("-o", "--output", type=str, default=None, help="Output image path. Otherwise show figure")
	parser.add_argument("--key", default="loss", help="Series to plot (loss, acc, time)")
	return parser.parse_args()


def main():
	print(f"DISPLAY: {os.getenv("DISPLAY", default="unset")}")
	print(f"MPL Backend: {matplotlib.get_backend()}")
	args = parse_args()

	if args.paths:
		logs = [get_logs(path) for path in args.paths]
	else:
		logs = [latest_logs_path()]

	SERIES_LABELS = {
		"loss_history": ("batch", "loss"),
		"acc_history": ("epoch", "accuracy"),
		"time_history": ("epoch", "time"),
	}

	key = args.key + "_history"
	(xlabel, ylabel) = SERIES_LABELS.get(key, ("step", "value"))

	fig, ax = plt.subplots()
	values = concat_logs(*logs, key=key)
	plot_logs(
		values,
		smoothing=args.smooth,
		ax=ax,
		xlabel=xlabel,
		ylabel=ylabel,
		label=key,
	)

	fig.tight_layout()

	if args.output:
		fig.savefig(args.output, bbox_inches="tight")
	else:
		plt.show()


if __name__ == '__main__':
	main()
