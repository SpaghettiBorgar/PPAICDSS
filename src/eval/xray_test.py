import argparse
import math
import os
import signal
import sys
from itertools import batched

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm

import data_prep
from models import xray_cnn
from models.xray_cnn import NORM_MEAN, NORM_STD
from training.xray.xray_data import XrayDataset, LABELS_SHORT as LABELS, TEST_OFFSET
from training.xray.xray_params import XrayParams
from util import numa
from util.mapping import tree_map
from util.utils import fix_collate


def print_pair_table(A: torch.Tensor, B: torch.Tensor, indices=None, labels=LABELS, print_rows=True, print_stats=True, file=sys.stdout):
	try:
		if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
			raise TypeError("A and B must be torch tensors.")
		if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 14:
			raise ValueError("A and B must both have shape (C, 14).")
		if len(A) == 0:
			raise ValueError("Can't use empty tensors.")
		if len(indices) != A.shape[0]:
			raise ValueError("indices must have length C.")
		if len(labels) != 14:
			raise ValueError("labels must have length 14.")
	except Exception as e:
		if isinstance(e, ValueError):
			raise e

	A = A.detach().cpu()
	B = B.detach().cpu()

	idx_w = max(len(str(x)) for x in indices + ["index"]) if indices is not None else 5

	a_w = 1
	b_w = 8

	# each pair column needs to fit at least the label
	col_w = [max(len(lbl), len(f"{0}") + 1 + b_w) for lbl in labels]

	norm_w = max(len("p1"), len("p2"), 10)

	header = [f"{'index':>{idx_w}}"]
	for lbl, w in zip(labels, col_w):
		header.append(f"{lbl:^{w}}")
	header.append(f"{'p1':>{norm_w}}")
	header.append(f"{'p2':>{norm_w}}")

	print(" | ".join(header), file=file)

	if print_rows:
		for i in range(len(A)):
			p1 = torch.norm(B[i] - A[i], p=1).item()
			p2 = torch.norm(B[i] - A[i], p=2).item()

			idx = indices[i] if indices is not None else ""
			row = [f"{idx:>{idx_w}}"]
			for j in range(14):
				a = int(A[i, j].item())
				b = B[i, j].item()
				cell = f"{a} {b:.4f}"
				row.append(f"{cell:>{col_w[j]}}")
			row.append(f"{p1:>{norm_w}.4f}")
			row.append(f"{p2:>{norm_w}.4f}")

			print(" | ".join(row), file=file)

	if not print_stats:
		return

	# ---Summary---

	diff = (B - A).abs()
	p1_vals = torch.norm(diff, p=1, dim=1)
	p2_vals = torch.norm(diff, p=2, dim=1)

	stat_names = ["min", "max", "std", "avg"]

	summary_rows = [
		("min", lambda x: torch.min(x, dim=0).values),
		("max", lambda x: torch.max(x, dim=0).values),
		("std", lambda x: torch.std(x, dim=0, unbiased=False)),
		("avg", lambda x: torch.mean(x, dim=0)),
	]

	for name, fn in summary_rows:
		row = [f"{name:>{idx_w}}"]

		vals = fn(diff)
		for j in range(14):
			row.append(f"{vals[j].item():>{col_w[j]}.4f}")

		p1_stat = fn(p1_vals)
		p2_stat = fn(p2_vals)

		row.append(f"{p1_stat.item():>{norm_w}.4f}")
		row.append(f"{p2_stat.item():>{norm_w}.4f}")

		print(" | ".join(row))

	data["p1"] = p1.numpy()
	data["p2"] = p2.numpy()

	df_new = pd.DataFrame(data)

	if os.path.exists(data_csv):
		df_old = pd.read_csv(data_csv)
		df_data = pd.concat([df_old, df_new], ignore_index=True)
	else:
		df_data = df_new

	df_data.to_csv(data_csv, index=False)

	# ------------------------------------------------------------
	# Summary CSV
	# Statistics are based on absolute difference |B - A|
	# ------------------------------------------------------------
	absdiff = torch.abs(diff)
	p1_abs = torch.norm(absdiff, p=1, dim=1)
	p2_abs = torch.norm(absdiff, p=2, dim=1)

	start = indices[0]
	end = indices[-1]
	n_rows = len(indices)

	stat_specs = [
		("min", lambda x: torch.min(x, dim=0).values),
		("max", lambda x: torch.max(x, dim=0).values),
		("std", lambda x: torch.std(x, dim=0, unbiased=False)),
		("avg", lambda x: torch.mean(x, dim=0)),
	]

	summary_rows = []
	for stat_name, stat_fn in stat_specs:
		row = {
			"start": start,
			"end": end,
			"n_rows": n_rows,
			"stat": stat_name,
		}

		vals = stat_fn(absdiff)
		for j, lbl in enumerate(labels):
			row[lbl] = vals[j].item()

		row["p1"] = stat_fn(p1_abs).item()
		row["p2"] = stat_fn(p2_abs).item()

		summary_rows.append(row)

	df_summary_new = pd.DataFrame(summary_rows)

	if os.path.exists(summary_csv):
		df_summary_old = pd.read_csv(summary_csv)
		df_summary = pd.concat([df_summary_old, df_summary_new], ignore_index=True)
	else:
		df_summary = df_summary_new

	df_summary.to_csv(summary_csv, index=False)


def binary_auroc(targets: torch.Tensor, scores: torch.Tensor) -> tuple[float, int, int, int]:
	targets = targets.detach().cpu().flatten()
	scores = scores.detach().cpu().flatten()

	valid = torch.isfinite(targets) & torch.isfinite(scores) & ((targets == 0) | (targets == 1))
	skipped = targets.numel() - valid.sum().item()
	targets = targets[valid]
	scores = scores[valid]

	n_pos = int((targets == 1).sum().item())
	n_neg = int((targets == 0).sum().item())
	if n_pos == 0 or n_neg == 0:
		return math.nan, n_pos, n_neg, skipped

	order = torch.argsort(scores)
	sorted_targets = targets[order]
	sorted_scores = scores[order]
	ranks = torch.empty(len(sorted_scores), dtype=torch.float64)

	start = 0
	while start < len(sorted_scores):
		end = start + 1
		while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
			end += 1
		ranks[start:end] = (start + 1 + end) / 2
		start = end

	pos_rank_sum = ranks[sorted_targets == 1].sum().item()
	auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
	return auc, n_pos, n_neg, skipped


def compute_per_class_aurocs(targets: torch.Tensor, outputs: torch.Tensor, labels=LABELS):
	if not isinstance(targets, torch.Tensor) or not isinstance(outputs, torch.Tensor):
		raise TypeError("targets and outputs must be torch tensors.")
	if targets.shape != outputs.shape:
		raise ValueError(f"targets and outputs must have the same shape, got {targets.shape} and {outputs.shape}.")
	if targets.ndim != 2 or targets.shape[1] != len(labels):
		raise ValueError(f"targets and outputs must both have shape (C, {len(labels)}), got {targets.shape}.")

	rows = []
	for j, label in enumerate(labels):
		auroc, n_pos, n_neg, skipped = binary_auroc(targets[:, j], outputs[:, j])
		rows.append({
			"label": label,
			"valid": n_pos + n_neg,
			"pos": n_pos,
			"neg": n_neg,
			"skipped": skipped,
			"auroc": auroc,
		})
	return rows


def print_auroc_table(targets: torch.Tensor, outputs: torch.Tensor, labels=LABELS):
	rows = compute_per_class_aurocs(targets, outputs, labels)
	label_w = max(len("class"), *(len(row["label"]) for row in rows))
	valid_w = max(len("valid"), *(len(str(row["valid"])) for row in rows))
	pos_w = max(len("pos"), *(len(str(row["pos"])) for row in rows))
	neg_w = max(len("neg"), *(len(str(row["neg"])) for row in rows))
	skipped_w = max(len("skipped"), *(len(str(row["skipped"])) for row in rows))
	auroc_w = len("AUROC")

	header = [
		f"{'class':>{label_w}}",
		f"{'valid':>{valid_w}}",
		f"{'pos':>{pos_w}}",
		f"{'neg':>{neg_w}}",
		f"{'skipped':>{skipped_w}}",
		f"{'AUROC':>{auroc_w}}",
	]
	print(" | ".join(header))

	valid_aurocs = []
	for row in rows:
		auroc = row["auroc"]
		if not math.isnan(auroc):
			valid_aurocs.append(auroc)
		auroc_str = "nan" if math.isnan(auroc) else f"{auroc:.4f}"
		print(" | ".join([
			f"{row['label']:>{label_w}}",
			f"{row['valid']:>{valid_w}}",
			f"{row['pos']:>{pos_w}}",
			f"{row['neg']:>{neg_w}}",
			f"{row['skipped']:>{skipped_w}}",
			f"{auroc_str:>{auroc_w}}",
		]))

	macro = math.nan if not valid_aurocs else sum(valid_aurocs) / len(valid_aurocs)
	macro_str = "nan" if math.isnan(macro) else f"{macro:.4f}"
	print(f"Macro AUROC: {macro_str}")


def test_xray_model(checkpoint, num_samples=0, print_samples: int | None = None, res: int | None = 512, crop: bool = True, device="cuda", offset=TEST_OFFSET, normalize=True):
	params = XrayParams(checkpoint=checkpoint, device=device)
	print(f"Testing model {params.checkpoint}", file=sys.stderr)
	model = params.get_model()
	transform = v2.Compose([
		*([v2.Resize(size=None, max_size=res, interpolation=InterpolationMode.BICUBIC)] if res is not None else []),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		*([v2.CenterCrop([res, res])] if crop else []),
		*([v2.Normalize(mean=[NORM_MEAN], std=[NORM_STD])] if normalize else [])
	])
	use_chunks = (res is not None and res <= data_prep.resolution and crop)
	print(f"use_chunks={use_chunks}", file=sys.stderr)
	dataset = XrayDataset(offset=offset, size=num_samples, use_chunks=use_chunks, transform=transform)

	do_quit = False

	def set_quit(*args):
		nonlocal do_quit
		print("Quitting", file=sys.stderr)
		do_quit = True

	signal.signal(signal.SIGINT, set_quit)
	signal.signal(signal.SIGTERM, set_quit)

	model.eval()
	torch.set_grad_enabled(False)

	all_targets = torch.empty((0, len(LABELS)))
	all_outputs = torch.empty((0, len(LABELS)))
	printed_samples = 0

	def record_batch(targets, outputs, real_idxs):
		nonlocal all_targets, all_outputs, printed_samples
		targets, outputs = targets.cpu(), outputs.cpu()
		rows_to_print = len(real_idxs) if print_samples is None else max(0, min(len(real_idxs), print_samples - printed_samples))
		if rows_to_print > 0:
			print(file=sys.stderr)
			print_pair_table(targets[:rows_to_print], outputs[:rows_to_print], real_idxs[:rows_to_print], file=sys.stderr)
			printed_samples += rows_to_print
		all_targets, all_outputs = torch.cat((all_targets, targets.clone())), torch.cat((all_outputs, outputs.clone()))

	if res is not None and crop and device != "cpu":
		print(f"Using batches on device {device}")
		loader = DataLoader(dataset, batch_size=16, shuffle=False,
		                    num_workers=int(os.getenv("DATA_LOADER_WORKERS", "2")), worker_init_fn=numa.WorkerInit(device))
		fix_collate(loader)
		for batch_idx, (inp, targets) in tqdm(enumerate(loader), file=sys.stderr):
			if do_quit:
				break
			inp, targets = tree_map(lambda t: t.to(device), (inp, targets))
			outputs = model(inp).sigmoid()
			start = batch_idx * loader.batch_size
			real_idxs = [dataset.real_index(i) for i in range(start, start + len(targets))]
			record_batch(targets, outputs, real_idxs)
	else:
		idxs = range(len(dataset))
		for idxs in tqdm(batched(idxs, 16), file=sys.stderr):
			if do_quit:
				break
			real_idxs = []
			targets = []
			outputs = []
			for i in idxs:
				if do_quit:
					break
				inp, target = dataset[i]
				inp, target = tree_map(lambda t: t.to(device).unsqueeze(0), (inp, target))
				output = model(inp).sigmoid()
				real_idxs.append(dataset.real_index(i))
				targets.append(target)
				outputs.append(output)

			targets, outputs = torch.cat(targets), torch.cat(outputs)
			record_batch(targets, outputs, real_idxs)

	print()
	print(" ".join(sys.argv))
	print()
	print_pair_table(all_targets, all_outputs, print_rows=False)
	print()
	print_auroc_table(all_targets, all_outputs)
	print()


def parse_args():
	parser = argparse.ArgumentParser(description="Train Xray Model")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use")
	parser.add_argument("--checkpoint", "-c", type=str, default="latest", help="Model checkpoint to load")
	parser.add_argument("--resolution", "-R", type=int, default=512, help="Max side length to scale images to")
	parser.add_argument("--crop", "-C", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=True, help="Crop/pad images to square")
	parser.add_argument("--num-samples", "-N", type=int, default=0, help="Number of samples to test")
	parser.add_argument("--print-samples", "-p", type=int, default=None, help="Number of tested samples to explicitly print")
	parser.add_argument("--offset", "-O", type=int, default=TEST_OFFSET, help="Sample index offset")
	parser.add_argument("--normalize", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=True, help="Normalize input images to resnet defaults")

	return parser.parse_args()


if __name__ == '__main__':
	print(f"cpu_count = {os.cpu_count()}, sched_affinity = {os.sched_getaffinity(0)}", file=sys.stderr)
	args = parse_args()
	if local_cpus := numa.device_local_cpus(args.device):
		# os.sched_setaffinity(0, local_cpus)
		print(f"Affinity now {os.sched_getaffinity(0)}", file=sys.stderr)
	print(f"interop_threads: {torch.get_num_interop_threads()}, num_threads: {torch.get_num_threads()}", file=sys.stderr)
	# torch.set_num_interop_threads(1)
	test_xray_model(args.checkpoint, num_samples=args.num_samples, print_samples=args.print_samples, res=args.resolution, crop=args.crop, device=args.device, offset=args.offset, normalize=args.normalize)
