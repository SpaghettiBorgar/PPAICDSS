import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.xray.xray_data import XrayDataset, CLASS_POS_WEIGHTS, TEST_OFFSET
from training.xray.xray_params import XrayParams
from util import numa
from util.mapping import tree_map
from util.utils import fix_collate


def parse_args():
	parser = argparse.ArgumentParser(description="Loss-based membership inference attack")
	parser.add_argument("--device", type=str, default="cuda", help="Device to use")
	parser.add_argument("--checkpoint", "-c", type=str, nargs="+", default=["latest"],
	                    help="One or more model checkpoints; each gets its own set of curves")
	parser.add_argument("--resolution", "-R", type=int, default=512, help="Max side length to scale images to")
	parser.add_argument("--num-samples", "-N", type=int, default=20000,
	                    help="Members and non-members each (capped at the held-out split size)")
	parser.add_argument("--batch-size", "-B", type=int, default=32, help="Eval batch size")
	parser.add_argument("--workers", "-w", type=int, default=int(os.getenv("DATA_LOADER_WORKERS", "0")),
	                    help="DataLoader workers for EACH dataset")
	parser.add_argument("--use-chunks", type=lambda x: x.lower() in ("true", "yes", "1"),
	                    choices=[True, False], default=True, help="Read pre-encoded zstd chunks")
	parser.add_argument("--percentile", "-p", type=float, default=90.0,
	                    help="Clip the threshold plots' x-axis to this percentile of loss values (100 = full range)")
	parser.add_argument("--output", "-o", type=str, default=None, help="Figure output path; show interactively if unset")
	parser.add_argument("--no-plot", action="store_true",
	                    help="Skip the figure and just print AUROC, best threshold, and accuracy/precision/recall there")
	parser.add_argument("--labels", "-l", type=str, nargs="+", default=None,
	                    help="Custom curve labels, in the same order as --checkpoint")
	parser.add_argument("--logx", action="store_true", help="Logarithmic threshold (x) axis on the three threshold plots")
	parser.add_argument("--latex", action="store_true",
	                    help="Keep figure text as real text (svg.fonttype=none, pdf/ps fonttype=42) so LaTeX can restyle fonts; save as .svg/.pdf")
	parser.add_argument("--csv", type=str, default=None,
	                    help="Export the plotted curve values (loss threshold, fpr, tpr, accuracy, precision, recall per model) to this CSV")

	args = parser.parse_args()
	if args.labels is not None and len(args.labels) != len(args.checkpoint):
		parser.error(f"--labels expects {len(args.checkpoint)} label(s) to match --checkpoint, got {len(args.labels)}")
	return args


@torch.no_grad()
def compute_losses(model, criterion, loader, device):
	fix_collate(loader)
	losses = []
	for inp, target in tqdm(loader, file=sys.stderr):
		inp, target = tree_map(lambda t: t.to(device), (inp, target))
		logits = model(inp)
		losses.append(criterion(logits, target).mean(dim=1).cpu())
	return torch.cat(losses).numpy()


def mia_curves(y_true, y_score, percentile=90.0):
	fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
	pos = float((y_true == 1).sum())
	neg = float((y_true == 0).sum())

	tp = tpr * pos
	fp = fpr * neg
	tn = neg - fp

	denom = tp + fp
	precision = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
	accuracy = (tp + tn) / (pos + neg)
	recall = tpr

	# roc_curve prepends an inf threshold (predict nothing positive); drop it and sort ascending
	# so the threshold plots read left-to-right over [min loss, max loss].
	finite = np.isfinite(thr)
	order = np.argsort(thr[finite])
	return dict(
		fpr=fpr, tpr=tpr, auc=roc_auc_score(y_true, y_score),
		thr=thr[finite][order],
		fpr_t=fpr[finite][order],  # fpr/tpr aligned to thr (for CSV); fpr/tpr above keep the ROC endpoints
		tpr_t=tpr[finite][order],
		accuracy=accuracy[finite][order],
		precision=precision[finite][order],
		recall=recall[finite][order],
		max_loss=float(np.max(y_score)),
		pct_loss=float(np.percentile(y_score, percentile)),
	)


def operating_point(r, xmax):
	"""Threshold of peak accuracy within the shown window, plus the metrics attained there.
	Every curve in `r` is indexed the same way, so one argmax reads all four values."""
	window = r["thr"] <= xmax
	if not window.any():
		return None
	best = np.argmax(np.where(window, r["accuracy"], -np.inf))
	return dict(thr=float(r["thr"][best]), accuracy=float(r["accuracy"][best]),
	            precision=float(r["precision"][best]), recall=float(r["recall"][best]))


def print_summary(results, percentile):
	xmax = max(r["pct_loss"] for r in results.values())
	for name, r in results.items():
		op = operating_point(r, xmax)
		print(f"\n{name}:")
		print(f"  MIA AUROC:              {r['auc']:.4f}")
		if op is not None:
			print(f"  best threshold:         {op['thr']:.4f}  (<= {percentile:g}th pct of loss)")
			print(f"  accuracy  @ threshold:  {op['accuracy']:.4f}")
			print(f"  precision @ threshold:  {op['precision']:.4f}")
			print(f"  recall    @ threshold:  {op['recall']:.4f}")


def _fmt(vals, spec):
	return [format(v, spec) if np.isfinite(v) else "" for v in vals]


def export_csv(results, path):
	"""One block of columns per model: loss_<label>, fpr<i>, tpr<i>, accuracy<i>, precision<i>, recall<i>.
	Loss thresholds in fixed 4-digit scientific notation, everything else fixed 4 decimals. Models with a
	different number of distinct thresholds are padded with blanks via an outer concat."""
	frames = []
	for i, (name, r) in enumerate(results.items()):
		frames.append(pd.DataFrame({
			f"loss_{name}": _fmt(r["thr"], ".4e"),
			f"fpr{i}": _fmt(r["fpr_t"], ".4f"),
			f"tpr{i}": _fmt(r["tpr_t"], ".4f"),
			f"accuracy{i}": _fmt(r["accuracy"], ".4f"),
			f"precision{i}": _fmt(r["precision"], ".4f"),
			f"recall{i}": _fmt(r["recall"], ".4f"),
		}))
	pd.concat(frames, axis=1).to_csv(path, index=False, na_rep="")
	print(f"Wrote CSV to {path}")


LEGEND_SETTINGS = dict(fontsize=8, framealpha=0.5, labelspacing=0.2, handlelength=1.5, handletextpad=0.6)


def plot_mia(results, output, logx=False):
	fig, (ax_roc, ax_acc, ax_prec, ax_rec) = plt.subplots(1, 4, figsize=(12, 3.0))

	for i, (name, r) in enumerate(results.items()):
		color = f"C{i}"
		ax_roc.plot(r["fpr"], r["tpr"], color=color, label=f"{name} (AUC={r['auc']:.3f})")
		ax_acc.plot(r["thr"], r["accuracy"], color=color, label=name)
		ax_prec.plot(r["thr"], r["precision"], color=color, label=name)
		ax_rec.plot(r["thr"], r["recall"], color=color, label=name)

	ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")  # chance diagonal
	ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC",
	           xlim=(0, 1), ylim=(0, 1))
	ax_roc.legend(loc="lower right", **LEGEND_SETTINGS)

	# Long-tailed losses stretch the axis, so clip to the widest per-model percentile cutoff.
	xmax = max(r["pct_loss"] for r in results.values())
	# log axis can't start at 0, so fall back to the smallest positive threshold shown
	positive = [t for r in results.values() for t in (r["thr"][r["thr"] > 0],) if t.size]
	xmin = min(t.min() for t in positive) if (logx and positive) else 0.0

	for i, (name, r) in enumerate(results.items()):
		color = f"C{i}"
		op = operating_point(r, xmax)  # peak accuracy is only meaningful within the shown range
		if op is not None:
			bt, ba = op["thr"], op["accuracy"]
			ax_acc.axvline(bt, color=color, ls=":", lw=0.8)
			ax_acc.plot([bt], [ba], marker="o", color=color, ms=5, zorder=5)
			# ax_acc.annotate(f"({bt:.2f}, {ba:.3f})", (bt, ba), textcoords="offset points",
			#                 xytext=(6, 6 + i * 14), fontsize=8, color=color)
			ax_acc.annotate(f"{ba:.3f}", (bt, ba), textcoords="offset points",
			                xytext=(6, 6 + i * 14), fontsize=8, color=color)

	for ax, title in ((ax_acc, "Accuracy"), (ax_prec, "Precision"), (ax_rec, "Recall")):
		ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=0)  # chance reference
		ax.set(xlabel="loss threshold", ylabel=title.lower(), title=title, ylim=(0, 1))
		if logx:
			ax.set_xscale("log")
		ax.set_xlim(xmin, xmax)
		ax.legend(**LEGEND_SETTINGS)

	fig.tight_layout()

	if output:
		fig.savefig(output, bbox_inches="tight")
		print(f"Saved figure to {output}")
	else:
		plt.show()


def main():
	print(f"DISPLAY: {os.getenv('DISPLAY', default='unset')}")
	print(f"MPL Backend: {matplotlib.get_backend()}")
	args = parse_args()

	if args.latex:
		# keep text as text (not paths) in vector output so the including document picks the font
		matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

	transform = XrayParams(phase="3", device=args.device, resolution=args.resolution).get_transform()
	n = min(args.num_samples, -TEST_OFFSET)
	inc = XrayDataset(transform=transform, offset=0, size=n, use_chunks=args.use_chunks)  # members (train split)
	exc = XrayDataset(transform=transform, offset=TEST_OFFSET, size=n, use_chunks=args.use_chunks)  # non-members (held-out)
	print(f"Members: {len(inc)}, non-members: {len(exc)}")

	workers = args.workers
	inc_loader = DataLoader(inc, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=True,
	                        worker_init_fn=numa.WorkerInit(args.device) if workers > 0 else None, persistent_workers=workers > 0)
	exc_loader = DataLoader(exc, batch_size=args.batch_size, shuffle=False, num_workers=workers, pin_memory=True,
	                        worker_init_fn=numa.WorkerInit(args.device) if workers > 0 else None, persistent_workers=workers > 0)

	criterion = nn.BCEWithLogitsLoss(pos_weight=CLASS_POS_WEIGHTS.to(args.device), reduction="none")
	y_true = np.concatenate([np.zeros(n), np.ones(n)])

	results = {}
	for idx, ckpt in enumerate(args.checkpoint):
		params = XrayParams(phase="3", checkpoint=ckpt, device=args.device, resolution=args.resolution)
		name = args.labels[idx] if args.labels else os.path.splitext(os.path.basename(params.checkpoint))[0]
		print(f"\n=== {name} ({params.checkpoint}) ===")
		model = params.get_model()
		model.eval()

		inc_losses = compute_losses(model, criterion, inc_loader, args.device)
		exc_losses = compute_losses(model, criterion, exc_loader, args.device)
		print(f"Member (train)  loss mean: {inc_losses.mean():.4f}")
		print(f"Non-member      loss mean: {exc_losses.mean():.4f}")

		y_score = np.concatenate([inc_losses, exc_losses])
		r = mia_curves(y_true, y_score, args.percentile)
		print(f"MIA AUROC: {r['auc']:.4f}")
		results[name] = r

	if args.csv:
		export_csv(results, args.csv)

	if args.no_plot:
		print_summary(results, args.percentile)
	else:
		plot_mia(results, args.output, logx=args.logx)


if __name__ == "__main__":
	main()
