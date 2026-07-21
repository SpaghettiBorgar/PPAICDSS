print("Initializing")
import argparse
import concurrent.futures
import dataclasses
import functools
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback
from typing import List, Callable, Any

import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.image import AxesImage
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from opacus.accountants.utils import get_noise_multiplier
from skimage.metrics import structural_similarity as sk_ssim
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.utils import make_grid

from models.simplenets import SimpleNet, LeNet1, LeNet3, SampleConvNet
from util.dp_compat import make_model_dp_compatible
from util.lazy import Lazy
from util.mapping import tree_map

os.environ.update({"PYCHARM_MATPLOTLIB_INTERACTIVE": "True"})

import matplotlib.pyplot as plt

import torchvision
from matplotlib.widgets import Slider
from torchvision.transforms import v2, InterpolationMode

from models.xray_cnn import XrayModel, NORM_MEAN, NORM_STD
from training.xray.xray_data import XrayDataset, DATA_DIR


def resnet_wrapper(block, layers, num_classes=None, weights=None, monochrome=False, make_smooth=False, **_):
	net = torchvision.models.resnet._resnet(block, layers, weights, progress=True)
	if monochrome:
		old_weights = net.conv1.weight.data
		net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		net.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)
		net.monochrome = monochrome
		if make_smooth:
			net.conv1.stride = (1, 1)
			net.relu = nn.SiLU(inplace=True)
	if num_classes is not None:
		net.fc = nn.Linear(net.fc.in_features, num_classes)
	return net


def densenet_wrapper(factory, num_classes=None, weights=None, **_):
	net = factory(weights=weights)
	if num_classes is not None:
		net.classifier = nn.Linear(net.classifier.in_features, num_classes)
	return net


OPTIMIZERS = {k: getattr(torch.optim, k) for k in torch.optim.__all__}
MODELS = {
			 cls.__name__: cls for cls in
			 [SimpleNet, LeNet1, LeNet3, SampleConvNet, XrayModel]
		 } | {
			 name: functools.partial(resnet_wrapper, block, layers, weights=weights, monochrome=monochrome, make_smooth=make_smooth)
			 for name, layers, block, weights, monochrome, make_smooth in [
		('ResNet0', [0, 0, 0, 0], BasicBlock, None, False, True),
		('ResNet0m', [0, 0, 0, 0], BasicBlock, None, True, True),
		('ResNet18', [2, 2, 2, 2], BasicBlock, torchvision.models.ResNet18_Weights.DEFAULT, False, False),
		('ResNet34', [3, 4, 6, 3], BasicBlock, torchvision.models.ResNet34_Weights.DEFAULT, False, False),
		('ResNet50', [3, 4, 6, 3], Bottleneck, torchvision.models.ResNet50_Weights.DEFAULT, False, False)
	]} | {
			 name: functools.partial(densenet_wrapper, factory, weights=None)
			 for name, factory, weights in [
		('DenseNet121', torchvision.models.densenet121, torchvision.models.DenseNet121_Weights.DEFAULT),
		('DenseNet169', torchvision.models.densenet169, torchvision.models.DenseNet169_Weights.DEFAULT),
		('DenseNet201', torchvision.models.densenet201, torchvision.models.DenseNet201_Weights.DEFAULT)
	]}
DATASETS = {
	'CXR': Lazy(XrayDataset, use_chunks=False),
	'CIFAR10': Lazy(torchvision.datasets.CIFAR10, root=f"{DATA_DIR}/torch_datasets", download=True),
	"MNIST": Lazy(torchvision.datasets.MNIST, root=f"{DATA_DIR}/torch_datasets", download=True)
}


def _process_img(img: Tensor):
	return make_grid(img.detach(), nrow=1, normalize=True).permute(1, 2, 0).cpu().clone()


def image_ssim(recovered: Tensor, target_processed) -> float | None:
	"""Structural similarity between a recovered image and the (pre-processed) target.

	Both are normalized to [0, 1] the same way they are displayed, so the score
	matches what is shown in the figure. Returns None if SSIM cannot be computed.
	"""
	rec = _process_img(recovered).numpy()
	tgt = target_processed
	if rec.shape != tgt.shape:
		return None
	height, width = rec.shape[:2]
	win = min(7, height, width)
	if win < 3:
		return None
	if win % 2 == 0:
		win -= 1
	try:
		if rec.ndim == 3 and rec.shape[-1] == 1:
			return float(sk_ssim(rec[..., 0], tgt[..., 0], data_range=1.0, win_size=win))
		return float(sk_ssim(rec, tgt, data_range=1.0, channel_axis=-1, win_size=win))
	except ValueError:
		return None


def format_ssim_value(ssim: float | None):
	if ssim is None or not math.isfinite(ssim):
		return None
	return f"{ssim:.3f}"


def format_loss_value(loss: float | None):
	if loss is None:
		return None
	if not math.isfinite(loss):
		return str(loss)
	return f"{loss:.3g}"


def format_loss_tick(value, _pos):
	if value <= 0 or not math.isfinite(value):
		return ""
	if 1e-2 <= value < 1e4:
		return f"{value:g}"
	return f"{value:.0e}"


class ImgPlot:
	gs: SubplotSpec
	subgs: GridSpec
	img_history: List[tuple[int, Tensor, float | None, float | None]]
	ax_img: Axes
	ax_loss: Axes | None
	ax_slider: Axes
	img: AxesImage
	loss_line: Any
	slider: Slider
	title: str
	summary: str | None
	loss_steps: list[int]
	loss_values: list[float]
	attempt_label: str | None
	ssim_text: Any

	def __init__(self, gs: SubplotSpec, init_img: Tensor, title="Recovered Image", loss: float | None = None,
	             show_loss=True, summary: str | None = None, ssim: float | None = None):
		self.gs = gs
		self.subgs = gs.subgridspec(3, 1, height_ratios=[3.0, 1.55, 0.28], hspace=0.55)
		self.title = title
		self.summary = summary
		self.loss_steps = []
		self.loss_values = []
		self.attempt_label = None

		self.img_history = [(0, _process_img(init_img), loss, ssim)]

		self.ax_img = plt.gcf().add_subplot(self.subgs[0])
		self.img = self.ax_img.imshow(self.img_history[0][1], aspect="equal")
		self.ax_img.set_box_aspect(1)
		self.ax_img.set_anchor("C")
		self.ax_img.set_axis_off()
		self.ssim_text = self.ax_img.text(0.5, -0.04, "", transform=self.ax_img.transAxes,
		                                  ha="center", va="top", fontsize=8)
		self._set_image_title(0)
		self._set_ssim_text(ssim)
		if summary:
			label = AnchoredText(summary, loc="lower left", prop={"size": 7}, frameon=True, borderpad=0.3)
			label.patch.set_alpha(0.75)
			label.set_clip_on(True)
			self.ax_img.add_artist(label)

		if show_loss:
			self.ax_loss = plt.gcf().add_subplot(self.subgs[1])
			self.loss_line, = self.ax_loss.plot([], [])
			self.ax_loss.set_xlabel("Iteration")
			self.ax_loss.set_yscale("log")
			self.ax_loss.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
			self.ax_loss.yaxis.set_major_formatter(FuncFormatter(format_loss_tick))
			self.ax_loss.yaxis.set_minor_formatter(NullFormatter())
			self.ax_loss.tick_params(axis="both", labelsize=8, pad=1)
			self._set_loss_title(loss)
		else:
			self.ax_loss = plt.gcf().add_subplot(self.subgs[1])
			self.ax_loss.set_axis_off()
			self.loss_line = None

		self.ax_slider = plt.gcf().add_subplot(self.subgs[-1])

		self.slider = Slider(self.ax_slider, 'Step', valmin=0, valmax=1, valinit=1, valstep=1)
		self.slider.on_changed(lambda val: self.update_plot())

	def _set_image_title(self, iter_num: int):
		label = f"i={iter_num}"
		if self.attempt_label:
			label = f"{self.attempt_label}, {label}"
		self.ax_img.set_title(f"{self.title} ({label})")

	def _set_loss_title(self, loss: float | None):
		if self.loss_line is None:
			return
		loss_text = format_loss_value(loss)
		self.ax_loss.set_title("Loss" if loss_text is None else f"Loss = {loss_text}", fontsize=9)

	def _set_ssim_text(self, ssim: float | None):
		ssim_text = format_ssim_value(ssim)
		self.ssim_text.set_text("" if ssim_text is None else f"SSIM = {ssim_text}")

	def update_plot(self):
		idx = min(int(self.slider.val), len(self.img_history) - 1)
		iter_num, img, loss, ssim = self.img_history[idx]
		self.img.set_data(img)
		self._set_image_title(iter_num)
		self._set_loss_title(loss)
		self._set_ssim_text(ssim)
		redraw_plot()

	def add_processed_img(self, img, idx, loss: float | None = None, ssim: float | None = None):
		self.img_history.append((idx, img, loss, ssim))
		slider = self.slider
		try:
			set_latest = slider.val >= slider.valmax
			slider.valmax = max(len(self.img_history) - 1, 1)
			slider.ax.set_xlim((slider.valmin, slider.valmax))
			if set_latest:
				slider.set_val(slider.valmax)
			else:
				self.update_plot()
		except NameError:
			pass

	def add_img(self, img, idx, loss: float | None = None, ssim: float | None = None):
		self.add_processed_img(_process_img(img), idx, loss, ssim)

	def load_recorded(self, recorded: list[tuple[int, Tensor, float | None, float | None]],
	                  loss_history: list[float] | None = None):
		if not recorded:
			return
		self.attempt_label = None
		self.img_history = [(step, _process_img(img), loss, ssim) for step, img, loss, ssim in recorded]

		self.loss_steps = []
		self.loss_values = []
		if self.loss_line is not None:
			self.loss_line.set_data([], [])
			if loss_history:
				self.add_losses(list(enumerate(loss_history, start=1)))
			else:
				self.add_losses([(step, loss) for step, _, loss, _ in recorded if loss is not None])

		slider = self.slider
		slider.valmax = max(len(self.img_history) - 1, 1)
		slider.ax.set_xlim((slider.valmin, slider.valmax))
		slider.set_val(slider.valmax)
		self.update_plot()

	def add_losses(self, points: list[tuple[int, float]]):
		if self.loss_line is None or not points:
			return

		for step, loss in points:
			if not math.isfinite(loss) or loss <= 0:
				continue
			self.loss_steps.append(step)
			self.loss_values.append(loss)
		if not self.loss_steps:
			return
		self.loss_line.set_data(self.loss_steps, self.loss_values)
		self._set_loss_title(self.loss_values[-1])
		self.ax_loss.relim()
		self.ax_loss.autoscale_view()

	def hide_slider(self):
		self.slider.set_active(False)
		self.slider.label.set_visible(False)
		self.slider.valtext.set_visible(False)
		self.ax_slider.set_axis_off()
		rect = plt.Rectangle((0, 0), 1, 1, transform=self.ax_slider.transAxes,
		                     color='white', zorder=10)
		self.ax_slider.add_patch(rect)


fig = None


def redraw_plot():
	fig.canvas.draw()
	fig.canvas.flush_events()


stop_event = threading.Event()


def handle_int(*args):
	if stop_event.is_set():
		sys.exit()
	else:
		stop_event.set()
		print("Stopping, press again to exit")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
	device: str = "cpu"
	max_iterations: int = 5000
	timeout: int = 15 * 60
	optimizer: str = "LBFGS"
	lr: float = 1.
	seed: int = 0
	resolution: int = 32
	model: str = "LeNet1"
	img: str = "CXR:0"
	target_mixin: float = 0.
	noise_mixin: float = 0.
	clipping_norm: float = 1.
	dp_noise: float = 0.
	dp_epsilon: float | None = None
	dp_delta: float = 1e-5
	dp_sample_rate: float = 1.
	dp_steps: int = 1
	dp_accountant: str = "rdp"
	weight_decay: float | None = None
	quantize: bool = False


def recover_image(model: nn.Module, gradients: List[Tensor], img: Tensor, labels: Tensor, criterion: nn.Module,
                  config: ExperimentConfig, *,
                  iter_callback: Callable[[int, Tensor, float], Any] | None = None,
                  rng: torch.Generator | None = None, stop: threading.Event | None = None, log_prefix=""):
	optimizer = OPTIMIZERS[config.optimizer]([img, labels], lr=config.lr, **({} if config.weight_decay is None else {'weight_decay': config.weight_decay}))
	# optimizer = OPTIMIZERS[config.optimizer]([img, ], lr=config.lr) # iDLG
	img.requires_grad = True
	labels.requires_grad = True
	# labels.requires_grad = False # iDLG

	i = 0
	start_time = time.time()
	stop = stop or stop_event
	while not (
			stop.is_set()
			or (config.max_iterations and i >= config.max_iterations)
			or (config.timeout and time.time() - start_time > config.timeout)
	):
		i += 1

		def closure():
			optimizer.zero_grad()

			pred_dummy = model(img)
			dummy_loss = criterion(pred_dummy, labels)

			dummy_grads = torch.autograd.grad(
				dummy_loss,
				model.parameters(),
				create_graph=True
			)

			grad_diff = 0
			for dg, rg in zip(dummy_grads, gradients):
				grad_diff += ((dg - rg) ** 2).sum()

			grad_diff.backward()

			return grad_diff

		prev_img = img.detach().clone()
		loss_val = optimizer.step(closure).item()
		diff_img = img.detach() - prev_img

		with torch.no_grad():
			img += torch.randn(img.shape, dtype=img.dtype, device=img.device, generator=rng) * config.noise_mixin

		if torch.isnan(img).any():
			print(f"{log_prefix}NAN")
			raise ValueError()

		if iter_callback is not None:
			iter_callback(i, img, loss_val)

		if i < 10 or i % 20 == 0:
			print(f"{log_prefix}Iteration {i}: Gradient Loss = {loss_val:.5f}, diff = {diff_img.min():.3E}/{diff_img.max():.3E}/{diff_img.std():.3E}")


def adapt_input_to_model(inp, model, transform=None):
	if type(inp) is tuple:
		assert len(inp) == 2 and isinstance(inp[0], torch.Tensor) and inp[0].ndim == 3
		img, x_view = inp
	else:
		img = inp

	if isinstance(model, XrayModel):
		channels = 1
	else:
		x_view = None
		if isinstance(model, (SimpleNet, SampleConvNet, LeNet1)) or getattr(model, "monochrome", False):
			channels = 1
		else:
			channels = 3

	if transform is not None:
		img = transform(img)

	if img.shape[0] == 1 and channels == 3:
		img = img.repeat(3, 1, 1)
	elif img.shape[0] == 3 and channels == 1:
		img = img.mean(dim=0, keepdim=True)

	return img

	if x_view is None:
		return img
	else:
		return img, x_view


def make_transform(resolution: int):
	return v2.Compose([
		v2.Resize(size=None, max_size=resolution, interpolation=InterpolationMode.BICUBIC),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.CenterCrop([resolution, resolution]),
		torch.nan_to_num,
		v2.Normalize(mean=[NORM_MEAN], std=[NORM_STD])
	])


def prepare_truth_tensors(model: nn.Module, inp, truth_label: Tensor, resolution: int, device="cpu"):
	truth_image = adapt_input_to_model(inp, model, make_transform(resolution))
	return tree_map(lambda t: t.unsqueeze(0).to(device), (truth_image, truth_label))


def resolve_dp_noise_multiplier(config: ExperimentConfig) -> float:
	if config.dp_epsilon is None:
		return config.dp_noise
	if config.dp_noise != 0:
		raise ValueError("Specify either --dp-noise or --dp-epsilon, not both")
	if config.dp_epsilon <= 0:
		raise ValueError("--dp-epsilon must be positive")
	if config.dp_delta <= 0 or config.dp_delta >= 1:
		raise ValueError("--dp-delta must be between 0 and 1")
	if config.dp_sample_rate <= 0 or config.dp_sample_rate > 1:
		raise ValueError("--dp-sample-rate must be in (0, 1]")
	if config.dp_steps < 1:
		raise ValueError("--dp-steps must be at least 1")
	return get_noise_multiplier(
		target_epsilon=config.dp_epsilon,
		target_delta=config.dp_delta,
		sample_rate=config.dp_sample_rate,
		steps=config.dp_steps,
		accountant=config.dp_accountant,
	)


def gradinversion(model: nn.Module, inp, truth_label: Tensor, config: ExperimentConfig, *, gs=None, gs_target=None,
                  plot_callback: Callable[[int, Tensor | None, float | None, float | None], Any] | None = None,
                  target_callback: Callable[[Tensor], Any] | None = None, run_name: str | None = None, **_):
	log_prefix = "" if run_name is None else f"[{run_name}] "
	model = make_model_dp_compatible(model).to(config.device)
	# criterion = nn.BCEWithLogitsLoss(pos_weight=xray_data.CLASS_POS_WEIGHTS.to(config.device))
	criterion = nn.BCEWithLogitsLoss()
	# criterion = nn.CrossEntropyLoss() # iDLG
	dp_noise = resolve_dp_noise_multiplier(config)

	truth_image, truth_label = prepare_truth_tensors(model, inp, truth_label, config.resolution, config.device)
	target_processed = _process_img(truth_image).numpy()
	rng = torch.Generator(device=truth_image.device)
	rng.manual_seed(config.seed)
	if target_callback is not None:
		target_callback(truth_image)

	if gs_target is not None:
		ImgPlot(gs_target, truth_image, title="Target Image", show_loss=False).hide_slider()

	recovered_image = torch.rand(truth_image.shape, dtype=truth_image.dtype, device=truth_image.device, generator=rng)
	print(f"{log_prefix}{truth_image.min()} {truth_image.max()} {recovered_image.min()} {recovered_image.max()}")
	recovered_image = (1. - config.target_mixin) * recovered_image + config.target_mixin * truth_image

	if config.dp_epsilon is not None:
		print(
			f"{log_prefix}DP target: epsilon={config.dp_epsilon:g}, delta={config.dp_delta:g}, "
			f"sample_rate={config.dp_sample_rate:g}, steps={config.dp_steps}, "
			f"accountant={config.dp_accountant}, sigma={dp_noise:g}"
		)

	initial_ssim = image_ssim(recovered_image, target_processed)
	plot = None if gs is None else ImgPlot(gs, recovered_image, title=run_name or "Recovered Image", ssim=initial_ssim)
	if plot_callback is not None:
		plot_callback(0, recovered_image, None, initial_ssim)

	model.eval()
	model.zero_grad()

	pred = model(truth_image)
	loss = criterion(pred, truth_label)
	print(f"{log_prefix}real loss: {loss}")

	real_grads = torch.autograd.grad(loss, model.parameters())
	real_grads = [g.detach().clone() for g in real_grads]
	flat_grads = torch.cat([g.flatten() for g in real_grads])
	grads_p2 = flat_grads.norm(p=2)
	print(f"actual gradient norm: {grads_p2}, min {flat_grads.min()}, max {flat_grads.max()}, std {flat_grads.std()}")
	real_grads = [g * min(1, config.clipping_norm / grads_p2) for g in real_grads]
	flat_grads = torch.cat([g.flatten() for g in real_grads])
	grads_p2 = flat_grads.norm(p=2)
	print(f"clipped gradient norm: {grads_p2}, min {flat_grads.min()}, max {flat_grads.max()}, std {flat_grads.std()}")
	# Apply DP noise
	real_grads = [
		g + torch.normal(mean=0, std=dp_noise * config.clipping_norm, size=g.shape, dtype=g.dtype, device=g.device, generator=rng)
		if not config.quantize else
		g + (torch.rand(size=g.shape, dtype=g.dtype, device=g.device, generator=rng) - 0.5) * (math.sqrt(12) * dp_noise * config.clipping_norm)
		for g in real_grads
	]
	with torch.no_grad():
		grad_flat = torch.cat([g.flatten() for g in real_grads])
		print(f"{log_prefix}real grads: {grad_flat.min():.3E}/{grad_flat.max():.3E}/{grad_flat.std():.3E}, p2={grad_flat.norm(p=2):.3E}")
		del grad_flat

	label_pred = torch.randn(truth_label.shape, dtype=truth_label.dtype, device=truth_label.device, generator=rng)
	# label_pred = torch.argmin(torch.sum(real_grads[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False) # iDLG
	# label_pred = one_hot(torch.argmin(torch.sum(real_grads[-2], dim=-1), dim=-1), truth_label.shape[-1]).to(torch.float32).detach().reshape((1,-1)).requires_grad_(False) # iDLG
	# print(f"{log_prefix}Target label: {truth_label}, Predicted label: {label_pred}")

	print(f"{log_prefix}Total model parameters: {sum([torch.numel(p) for p in model.parameters()])}")
	print(f"{log_prefix}Total gradinversion parameters: {sum([torch.numel(recovered_image), torch.numel(label_pred)])}")

	loss_history = []

	def callback(i, img, loss):
		loss_history.append(loss)
		should_add_image = i < 10 or (i < 1000 and i % 20 == 0) or i % 100 == 0
		ssim_val = image_ssim(img, target_processed) if should_add_image else None
		if plot_callback is not None:
			plot_callback(i, img if should_add_image else None, loss, ssim_val)
		if plot is not None:
			plot.add_losses([(i, loss)])
			if should_add_image:
				plot.add_img(img, i, loss, ssim_val)
				redraw_plot()

	recover_image(model, real_grads, recovered_image, label_pred, criterion,
	              config=config, iter_callback=callback, rng=rng, stop=stop_event, log_prefix=log_prefix)
	# recover_image(model, real_grads, recovered_image, truth_label, criterion,
	#   config=config, iter_callback=callback, rng=rng, stop=stop_event, log_prefix=log_prefix)

	final_ssim = image_ssim(recovered_image, target_processed)
	print(f"{log_prefix}Final SSIM = {format_ssim_value(final_ssim) or 'n/a'}")
	return loss_history, final_ssim


def load_sample(path):
	splits = path.split(':')
	return DATASETS[splits[0]][splits[1]]


CONFIG_LABELS = {
	"device": "dev",
	"max_iterations": "iters",
	"timeout": "timeout",
	"optimizer": "opt",
	"lr": "lr",
	"seed": "seed",
	"resolution": "R",
	"model": "model",
	"img": "img",
	"target_mixin": "target-mix",
	"noise_mixin": "noise",
	"clipping_norm": "C",
	"dp_noise": "D",
	"dp_epsilon": "ε",
	"dp_delta": "δ",
	"dp_sample_rate": "q",
	"dp_steps": "steps",
	"dp_accountant": "acct",
	"weight_decay": "w_decay",
	"quantize": "quantize"
}

DEFAULT_LABEL_FIELDS = (
	"model", "optimizer", "resolution", "target_mixin", "noise_mixin", "clipping_norm",
	"dp_noise", "dp_epsilon", "dp_delta", "dp_sample_rate", "dp_steps", "dp_accountant",
	"weight_decay", "quantize"
)


def parse_label_fields(spec: str | None) -> tuple[str, ...]:
	if spec is None:
		return DEFAULT_LABEL_FIELDS
	value = spec.strip()
	if value.lower() in {"", "none"}:
		return ()
	if value.lower() == "all":
		return tuple(CONFIG_LABELS)
	fields: list[str] = []
	unknown: list[str] = []
	for part in value.split(','):
		key = part.strip().replace('-', '_')
		if not key:
			continue
		if key not in CONFIG_LABELS:
			unknown.append(part.strip())
		elif key not in fields:
			fields.append(key)
	if unknown:
		raise ValueError(
			f"Unknown label field(s): {', '.join(unknown)}. Available: {', '.join(CONFIG_LABELS)}"
		)
	return tuple(fields)


def format_config_value(value: Any):
	if isinstance(value, float):
		return f"{value:g}"
	return str(value)


def format_config_summary(config: ExperimentConfig, baselines=(ExperimentConfig(),), *,
                          labels: tuple[str, ...] | None = None, max_items=6, per_line=2):
	label_fields = DEFAULT_LABEL_FIELDS if labels is None else labels
	items = []
	for field in dataclasses.fields(ExperimentConfig):
		name = field.name
		if name not in label_fields or name not in CONFIG_LABELS:
			continue
		value = getattr(config, name)
		if any(value != getattr(baseline, name) for baseline in baselines):
			label = CONFIG_LABELS.get(name, name.replace("_", "-"))
			items.append(f"{label}={format_config_value(value)}")

	if not items:
		return "defaults"

	visible = items[:max_items]
	if len(items) > max_items:
		visible.append(f"+{len(items) - max_items}")

	return "\n".join(
		", ".join(visible[i:i + per_line])
		for i in range(0, len(visible), per_line)
	)


@dataclasses.dataclass(frozen=True)
class ExperimentSpec:
	idx: int
	name: str
	config: ExperimentConfig
	baseline: ExperimentConfig


@dataclasses.dataclass(frozen=True)
class ParsedArgs:
	experiments: list[ExperimentSpec]
	workers: int
	output: str | None
	best_of: int
	label_fields: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RunUpdate:
	run_idx: int
	step: int
	loss: float | None
	img: Tensor | None = None
	ssim: float | None = None
	attempt: str | None = None


@dataclasses.dataclass(frozen=True)
class DoneUpdate:
	run_idx: int
	loss_history: list[float]
	ssim: float | None = None
	recorded: list[tuple[int, Tensor, float | None, float | None]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ErrorUpdate:
	run_idx: int
	error: BaseException
	trace: str


def add_config_args(parser: argparse.ArgumentParser, *, defaults: bool):
	default = None if defaults else argparse.SUPPRESS
	parser.add_argument("--device", default=default, help="Device to use")
	parser.add_argument("--max-iterations", "-I", type=int, default=default, help="Maximum iterations (0 for unlimited)")
	parser.add_argument("--timeout", type=int, default=default, help="Maximum time (s)")
	parser.add_argument("--optimizer", "--opt", choices=OPTIMIZERS, default=default)
	parser.add_argument("--lr", "--learning-rate", type=float, default=default)
	parser.add_argument("--seed", type=int, default=default)
	parser.add_argument("--resolution", "-R", type=int, default=default)
	parser.add_argument("--model", choices=MODELS, default=default)
	parser.add_argument("--img", default=default, help=f"Input sample to recreate (dataset:idx). Available datasets: {{{', '.join(DATASETS.keys())}}}")
	parser.add_argument("--target-mixin", type=float, default=default, help="Percentage of target to mix into starting image")
	parser.add_argument("--noise-mixin", "-N", type=float, default=default, help="Noise amplitude to mix in each iteration")
	parser.add_argument("--clipping-norm", "-C", type=float, default=default, help="Gradient clipping norm for DP")
	parser.add_argument("--dp-noise", "-D", type=float, default=default, help="Noise multiplier for DP")
	parser.add_argument("--dp-epsilon", "--target-epsilon", "--eps", type=float, default=default, help="Target epsilon for Opacus-calibrated DP noise")
	parser.add_argument("--dp-delta", "--target-delta", type=float, default=default, help="Target delta for Opacus-calibrated DP noise")
	parser.add_argument("--dp-sample-rate", type=float, default=default, help="Sample rate used by the Opacus accountant")
	parser.add_argument("--dp-steps", type=int, default=default, help="Number of DP steps used by the Opacus accountant")
	parser.add_argument("--dp-accountant", choices=["rdp", "gdp", "prv"], default=default, help="Opacus accountant mechanism")
	parser.add_argument("--weight-decay", type=float, default=default, help="Optimizer weight decay")
	parser.add_argument("--quantize", type=lambda x: x.lower() in ['true', 'yes'], default=default, help="Attempt equivalent perturbation using quantization noise")
	parser.add_argument("--name", default=default, help="Display name for this run")


def make_config_parser(add_help=False):
	parser = argparse.ArgumentParser(add_help=add_help)
	add_config_args(parser, defaults=False)
	parser.add_argument("--workers", type=int, default=argparse.SUPPRESS, help="Maximum worker threads (default: one per run)")
	parser.add_argument("--output", "-o", default=argparse.SUPPRESS, help="Save the final visualization to this image file")
	parser.add_argument("--best-of", type=int, default=argparse.SUPPRESS, help="Repeat each experiment this many times (seed incremented each time) and keep the run with the best SSIM")
	parser.add_argument("--labels", default=argparse.SUPPRESS, help="Comma-separated config fields to show as labels ('all', 'none', or e.g. model,lr,seed)")
	return parser


def make_help_parser():
	parser = argparse.ArgumentParser(
		description="Attempt gradient inversion attacks, optionally running multiple concurrent experiments.",
		epilog=(
			"Options outside --run mutate the current baseline. --run creates an experiment from that "
			"baseline plus comma-separated overrides that do not affect later runs. Example: "
			"--seed 0 --lr 0.1 --opt Adam --run seed=1 --run seed=2 --opt SGD --run lr=0.01. "
			"--best-of N repeats each experiment N times with incrementing seeds and keeps the run "
			"with the highest SSIM to the target."
		)
	)
	add_config_args(parser, defaults=True)
	parser.set_defaults(**dataclasses.asdict(ExperimentConfig()), name=None)
	parser.add_argument("--workers", type=int, default=None, help="Maximum worker threads (default: one per run)")
	parser.add_argument("--output", "-o", default=None, help="Save the final visualization to this image file")
	parser.add_argument("--best-of", type=int, default=1, help="Repeat each experiment this many times (seed incremented each time) and keep the run with the best SSIM")
	parser.add_argument("--labels", default=None,
	                    help=f"Comma-separated config fields to show as labels ('all', 'none', or a subset of: {', '.join(CONFIG_LABELS)})")
	parser.add_argument("--run", "--experiment", nargs="?", metavar="OVERRIDES",
	                    help="Create an experiment from the current baseline plus overrides like seed=1,lr=0.1")
	return parser


def expand_key_value_args(tokens: list[str]):
	expanded: list[str] = []
	for token in tokens:
		if not token.startswith("-") and "=" in token:
			key, value = token.split("=", 1)
			expanded.extend([f"--{key.replace('_', '-')}", value])
		else:
			expanded.append(token)
	return expanded


def expand_run_spec(spec: str | None):
	if spec is None:
		return None, []

	name = None
	args: list[str] = []
	for part in (p.strip() for p in spec.split(',')):
		if not part:
			continue
		if "=" not in part:
			name = part
			continue
		key, value = part.split("=", 1)
		if key == "name":
			name = value
		else:
			args.extend([f"--{key.replace('_', '-')}", value])
	return name, args


def parse_args(argv: list[str] | None = None):
	argv = sys.argv[1:] if argv is None else argv
	if any(token in {"-h", "--help"} for token in argv):
		make_help_parser().parse_args(["--help"])

	parser = make_config_parser()
	config = ExperimentConfig()
	next_name: str | None = None
	workers: int | None = None
	output: str | None = None
	best_of: int = 1
	labels_spec: str | None = None
	experiments: list[ExperimentSpec] = []
	pending_tokens: list[str] = []

	def apply_pending():
		nonlocal config, next_name, workers, output, best_of, labels_spec, pending_tokens
		if not pending_tokens:
			return
		overrides = vars(parser.parse_args(expand_key_value_args(pending_tokens)))
		pending_tokens = []
		if "workers" in overrides:
			workers = overrides.pop("workers")
		if "output" in overrides:
			output = overrides.pop("output")
		if "best_of" in overrides:
			best_of = overrides.pop("best_of")
		if "labels" in overrides:
			labels_spec = overrides.pop("labels")
		if "name" in overrides:
			next_name = overrides.pop("name")
		config = dataclasses.replace(config, **overrides)

	i = 0
	while i < len(argv):
		token = argv[i]
		if token in {"--run", "--experiment"}:
			apply_pending()
			run_spec = None
			if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
				run_spec = argv[i + 1]
				i += 1
			run_name, run_args = expand_run_spec(run_spec)
			run_overrides = vars(parser.parse_args(run_args))
			if any(k in run_overrides for k in ("workers", "output", "best_of", "labels")):
				parser.error("--workers, --output, --best-of and --labels are global options and cannot be used inside --run overrides")
			run_config = dataclasses.replace(config, **run_overrides)
			name = run_name or next_name or f"run {len(experiments) + 1}"
			experiments.append(ExperimentSpec(len(experiments), name, run_config, config))
			next_name = None
		else:
			pending_tokens.append(token)
		i += 1

	apply_pending()

	if not experiments:
		experiments.append(ExperimentSpec(0, next_name or "run 1", config, ExperimentConfig()))

	if workers is None:
		workers = len(experiments)
	try:
		label_fields = parse_label_fields(labels_spec)
	except ValueError as e:
		parser.error(str(e))
	return ParsedArgs(experiments, workers=max(1, workers), output=output, best_of=max(1, best_of),
	                  label_fields=label_fields)


def load_experiment_components(config: ExperimentConfig):
	img_split = config.img.split(':')
	if len(img_split) != 2 or img_split[0] not in DATASETS:
		raise ValueError(f"Invalid image spec {config.img!r}; expected dataset:idx")
	dataset = DATASETS[img_split[0]]
	inp, target_ = dataset[int(img_split[1])]
	if type(target_) is int:
		target_ = one_hot(torch.tensor(target_), len(dataset.classes)).to(dtype=torch.float32)
	# target = torch.zeros_like(target_)
	# target[torch.nonzero(target_)[0]] = 1. # convert to single-label classification
	target = target_
	model = MODELS[config.model](num_classes=len(dataset.classes), resolution=config.resolution)
	return model, inp, target


def run_experiment(spec: ExperimentSpec, updates: queue.Queue[RunUpdate | DoneUpdate | ErrorUpdate], best_of: int = 1):
	try:
		best_of = max(1, best_of)
		suffix = "" if best_of == 1 else f" (best of {best_of})"
		print(f"[{spec.name}] Running with config: {spec.config}{suffix}")

		have_best = False
		best_ssim: float | None = None
		best_recorded: list[tuple[int, Tensor, float | None, float | None]] = []
		best_loss_history: list[float] = []

		for attempt in range(best_of):
			attempt_config = spec.config if best_of == 1 else dataclasses.replace(spec.config, seed=spec.config.seed + attempt)
			attempt_label = None if best_of == 1 else f"try {attempt + 1}/{best_of}"
			model, inp, target = load_experiment_components(attempt_config)
			recorded: list[tuple[int, Tensor, float | None, float | None]] = []

			def plot_callback(step: int, img: Tensor | None, loss: float | None, ssim: float | None,
			                  _recorded=recorded, _label=attempt_label):
				plot_img = None if img is None else img.detach().cpu().clone()
				if plot_img is not None:
					_recorded.append((step, plot_img, loss, ssim))
				updates.put(RunUpdate(spec.idx, step, loss, plot_img, ssim, _label))

			loss_history, final_ssim = gradinversion(
				model=model,
				inp=inp,
				truth_label=target,
				config=attempt_config,
				plot_callback=plot_callback,
				run_name=spec.name if best_of == 1 else f"{spec.name} {attempt_label}"
			)

			run_ssim = recorded[-1][3] if recorded else final_ssim
			is_better = not have_best or (run_ssim is not None and (best_ssim is None or run_ssim > best_ssim))
			if is_better:
				have_best = True
				best_ssim = run_ssim
				best_recorded = recorded
				best_loss_history = loss_history

		updates.put(DoneUpdate(spec.idx, best_loss_history, best_ssim, best_recorded))
	except Exception as e:
		updates.put(ErrorUpdate(spec.idx, e, traceback.format_exc()))


def make_target_image(spec: ExperimentSpec):
	model, inp, target = load_experiment_components(spec.config)
	truth_image, _ = prepare_truth_tensors(model, inp, target, spec.config.resolution, device="cpu")
	return truth_image


def process_plot_updates(updates: list[RunUpdate | DoneUpdate | ErrorUpdate], run_plots: dict[int, ImgPlot],
                         loss_histories: dict[int, list[float]], errors: list[ErrorUpdate],
                         grid: GridSpec, specs: list[ExperimentSpec], label_fields: tuple[str, ...] = DEFAULT_LABEL_FIELDS):
	loss_batches: dict[int, list[tuple[int, float]]] = {}
	image_updates: dict[int, RunUpdate] = {}
	done_updates: dict[int, DoneUpdate] = {}

	for update in updates:
		match update:
			case RunUpdate(run_idx=run_idx, step=step, loss=loss, img=img):
				if loss is not None:
					loss_batches.setdefault(run_idx, []).append((step, loss))
				if img is not None:
					image_updates[run_idx] = update
			case DoneUpdate(run_idx=run_idx, loss_history=loss_history, ssim=ssim) as done:
				loss_histories[run_idx] = loss_history
				done_updates[run_idx] = done
				ssim_text = format_ssim_value(ssim) or "n/a"
				print(f"[{specs[run_idx].name}] Finished after {len(loss_history)} optimization steps (best SSIM = {ssim_text})")
			case ErrorUpdate(run_idx=run_idx) as error:
				errors.append(error)
				print(f"[{specs[run_idx].name}] Failed: {error.error}")

	for spec in specs:
		plot = run_plots.get(spec.idx)
		img_update = image_updates.get(spec.idx)
		done = done_updates.get(spec.idx)
		created_plot = False

		init_img = init_loss = init_ssim = None
		if img_update is not None:
			init_img, init_loss, init_ssim = img_update.img, img_update.loss, img_update.ssim
		elif done is not None and done.recorded:
			_, init_img, init_loss, init_ssim = done.recorded[0]

		if plot is None and init_img is not None:
			plot = ImgPlot(
				grid[0, spec.idx + 1], init_img,
				title=spec.name, loss=init_loss, ssim=init_ssim,
				summary=format_config_summary(spec.config, baselines=(ExperimentConfig(), spec.baseline), labels=label_fields)
			)
			run_plots[spec.idx] = plot
			created_plot = True
		if plot is None:
			continue
		plot.add_losses(loss_batches.get(spec.idx, []))
		if img_update is not None and not created_plot:
			plot.attempt_label = img_update.attempt
			plot.add_img(img_update.img, img_update.step, img_update.loss, img_update.ssim)
		if done is not None:
			plot.load_recorded(done.recorded, done.loss_history)


def make_figure(num_experiments: int):
	num_columns = num_experiments + 1
	column_width = 2.35
	fig_width = max(6.8, column_width * num_columns + 0.7)
	fig_height = 4.9
	figure = plt.figure(figsize=(fig_width, fig_height), dpi=100)
	figure.subplots_adjust(left=0.035, right=0.99, bottom=0.08, top=0.92)
	grid = figure.add_gridspec(1, num_columns, wspace=0.34)

	try:
		manager = plt.get_current_fig_manager()
		if hasattr(manager, "window") and hasattr(manager.window, "resizable"):
			manager.window.resizable(False, False)
	except Exception:
		pass

	return figure, grid


def main():
	parsed = parse_args()
	specs = parsed.experiments

	signal.signal(signal.SIGINT, handle_int)

	plt.ion()
	global fig
	fig, grid = make_figure(len(specs))
	target_plot = ImgPlot(grid[0, 0], make_target_image(specs[0]), title="Target Image", show_loss=False)
	target_plot.hide_slider()
	redraw_plot()

	updates: queue.Queue[RunUpdate | DoneUpdate | ErrorUpdate] = queue.Queue()
	run_plots: dict[int, ImgPlot] = {}
	loss_histories: dict[int, list[float]] = {}
	errors: list[ErrorUpdate] = []

	best_of_note = "" if parsed.best_of == 1 else f", best of {parsed.best_of} per experiment"
	print(f"Starting {len(specs)} experiment(s) on {parsed.workers} worker thread(s){best_of_note}")
	with concurrent.futures.ThreadPoolExecutor(max_workers=parsed.workers, thread_name_prefix="gradinv") as executor:
		futures = [executor.submit(run_experiment, spec, updates, parsed.best_of) for spec in specs]

		while True:
			batch: list[RunUpdate | DoneUpdate | ErrorUpdate] = []
			try:
				batch.append(updates.get(timeout=0.05))
				while True:
					batch.append(updates.get_nowait())
			except queue.Empty:
				pass

			if batch:
				process_plot_updates(batch, run_plots, loss_histories, errors, grid, specs, parsed.label_fields)
				redraw_plot()

			plt.pause(0.05)
			if all(f.done() for f in futures) and updates.empty():
				break

		for future in futures:
			future.result()

	for error in errors:
		print(error.trace, file=sys.stderr)

	if parsed.output is not None:
		target_plot.hide_slider()
		for plot in run_plots.values():
			plot.hide_slider()
		redraw_plot()
		output_dir = os.path.dirname(parsed.output)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		import subprocess
		print(subprocess.getoutput('date'))
		fig.savefig(parsed.output, dpi=300)
		print(f"Saved visualization to {parsed.output}")
		sys.exit()
		print(f"Couldn't exit")

	plt.ioff()
	plt.show()


if __name__ == '__main__':
	main()
