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
from opacus.accountants.utils import get_noise_multiplier
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.image import AxesImage
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.utils import make_grid
from matplotlib.offsetbox import AnchoredText

from util.dp_compat import make_model_dp_compatible
from models.simplenets import SimpleNet, LeNet1, LeNet3, SampleConvNet
from util.lazy import Lazy
from util.mapping import tree_map

os.environ.update({"PYCHARM_MATPLOTLIB_INTERACTIVE": "True"})

import matplotlib.pyplot as plt

import torchvision
from matplotlib.widgets import Slider
from torchvision.transforms import v2, InterpolationMode

from models.xray_cnn import XrayModel
from training.xray.xray_data import XrayDataset, DATA_DIR


def resnet_wrapper(block, layers, num_classes=None, weights=None, **_):
	net = torchvision.models.resnet._resnet(block, layers, weights, progress=True)
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
		name: functools.partial(resnet_wrapper, block, layers, weights=None)
		for name, layers, block, weights in [
		('ResNet0', [0, 0, 0, 0], BasicBlock, None),
		('ResNet18', [2, 2, 2, 2], BasicBlock, torchvision.models.ResNet18_Weights.DEFAULT),
		('ResNet34', [3, 4, 6, 3], BasicBlock, torchvision.models.ResNet34_Weights.DEFAULT),
		('ResNet50', [3, 4, 6, 3], Bottleneck, torchvision.models.ResNet50_Weights.DEFAULT)
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
	img_history: List[tuple[int, Tensor, float | None]]
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

	def __init__(self, gs: SubplotSpec, init_img: Tensor, title="Recovered Image", loss: float | None = None,
	             show_loss=True, summary: str | None = None):
		self.gs = gs
		self.subgs = gs.subgridspec(3, 1, height_ratios=[3.0, 1.55, 0.28], hspace=0.55)
		self.title = title
		self.summary = summary
		self.loss_steps = []
		self.loss_values = []

		self.img_history = [(0, _process_img(init_img), loss)]

		self.ax_img = plt.gcf().add_subplot(self.subgs[0])
		self.img = self.ax_img.imshow(self.img_history[0][1], aspect="equal")
		self.ax_img.set_box_aspect(1)
		self.ax_img.set_anchor("C")
		self.ax_img.set_axis_off()
		self._set_image_title(0)
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
		self.ax_img.set_title(f"{self.title} (i={iter_num})")

	def _set_loss_title(self, loss: float | None):
		if self.loss_line is None:
			return
		loss_text = format_loss_value(loss)
		self.ax_loss.set_title("Loss" if loss_text is None else f"Loss = {loss_text}", fontsize=9)

	def update_plot(self):
		idx = min(int(self.slider.val), len(self.img_history) - 1)
		iter_num, img, loss = self.img_history[idx]
		self.img.set_data(img)
		self._set_image_title(iter_num)
		self._set_loss_title(loss)
		redraw_plot()

	def add_processed_img(self, img, idx, loss: float | None = None):
		self.img_history.append((idx, img, loss))
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

	def add_img(self, img, idx, loss: float | None = None):
		self.add_processed_img(_process_img(img), idx, loss)

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
	lr: float = 0.01
	seed: int = 0
	resolution: int = 32
	model: str = "LeNet"
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


def recover_image(model: nn.Module, gradients: List[Tensor], img: Tensor, labels: Tensor, criterion: nn.Module,
                  config: ExperimentConfig, *,
                  iter_callback: Callable[[int, Tensor, float], Any] | None = None,
                  rng: torch.Generator | None = None, stop: threading.Event | None = None, log_prefix=""):
	optimizer = OPTIMIZERS[config.optimizer]([img, labels], lr=config.lr)
	img.requires_grad = True
	labels.requires_grad = True

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
		if isinstance(model, (SimpleNet, SampleConvNet, LeNet1)):
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
		torch.nan_to_num
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
				  plot_callback: Callable[[int, Tensor | None, float | None], Any] | None = None,
				  target_callback: Callable[[Tensor], Any] | None = None, run_name: str | None = None, **_):
	log_prefix = "" if run_name is None else f"[{run_name}] "
	model = make_model_dp_compatible(model).to(config.device)
	# criterion = nn.BCEWithLogitsLoss(pos_weight=xray_params.CLASS_WEIGHTS.to(device))
	criterion = nn.BCEWithLogitsLoss()
	dp_noise = resolve_dp_noise_multiplier(config)

	truth_image, truth_label = prepare_truth_tensors(model, inp, truth_label, config.resolution, config.device)
	rng = torch.Generator(device=truth_image.device)
	rng.manual_seed(config.seed)
	if target_callback is not None:
		target_callback(truth_image)

	if gs_target is not None:
		ImgPlot(gs_target, truth_image, title="Target Image", show_loss=False).hide_slider()

	recovered_image = torch.rand(truth_image.shape, dtype=truth_image.dtype, device=truth_image.device, generator=rng)
	print(f"{log_prefix}{truth_image.min()} {truth_image.max()} {recovered_image.min()} {recovered_image.max()}")
	recovered_image = (1. - config.target_mixin) * recovered_image + config.target_mixin * truth_image

	recovered_label_logits = torch.randn(truth_label.shape, dtype=truth_label.dtype, device=truth_label.device, generator=rng)

	print(f"{log_prefix}Total model parameters: {sum([torch.numel(p) for p in model.parameters()])}")
	print(f"{log_prefix}Total gradinversion parameters: {sum([torch.numel(recovered_image), torch.numel(recovered_label_logits)])}")
	if config.dp_epsilon is not None:
		print(
			f"{log_prefix}DP target: epsilon={config.dp_epsilon:g}, delta={config.dp_delta:g}, "
			f"sample_rate={config.dp_sample_rate:g}, steps={config.dp_steps}, "
			f"accountant={config.dp_accountant}, sigma={dp_noise:g}"
		)

	plot = None if gs is None else ImgPlot(gs, recovered_image, title=run_name or "Recovered Image")
	if plot_callback is not None:
		plot_callback(0, recovered_image, None)

	model.eval()
	model.zero_grad()

	pred = model(truth_image)
	loss = criterion(pred, truth_label)
	print(f"{log_prefix}real loss: {loss}")

	real_grads = torch.autograd.grad(loss, model.parameters())
	real_grads = [g.detach().clone() for g in real_grads]
	grads_p2 = torch.cat([g.flatten() for g in real_grads]).norm(p=2)
	real_grads = [g * min(1, config.clipping_norm / grads_p2) for g in real_grads]
	# Apply DP noise
	real_grads = [
		g + torch.normal(mean=0, std=dp_noise * config.clipping_norm, size=g.shape, dtype=g.dtype, device=g.device, generator=rng)
		for g in real_grads
	]

	loss_history = []

	def callback(i, img, loss):
		loss_history.append(loss)
		should_add_image = i < 10 or (i < 1000 and i % 20 == 0) or i % 100 == 0
		if plot_callback is not None:
			plot_callback(i, img if should_add_image else None, loss)
		if plot is not None:
			plot.add_losses([(i, loss)])
			if should_add_image:
				plot.add_img(img, i)
				redraw_plot()

	recover_image(model, real_grads, recovered_image, recovered_label_logits, criterion,
	              config=config, iter_callback=callback, rng=rng, stop=stop_event, log_prefix=log_prefix)

	return loss_history


def load_sample(path):
	splits = path.split(':')
	return DATASETS[splits[0]][splits[1]]


CONFIG_LABELS = {
	"model": "model",
	"optimizer": "opt",
	"resolution": "R",
	"target_mixin": "target-mix",
	"noise_mixin": "noise",
	"clipping_norm": "C",
	"dp_noise": "D",
	"dp_epsilon": "ε",
	"dp_delta": "δ",
	"dp_sample_rate": "q",
	"dp_steps": "steps",
	"dp_accountant": "acct",
}


def format_config_value(value: Any):
	if isinstance(value, float):
		return f"{value:g}"
	return str(value)


def format_config_summary(config: ExperimentConfig, baselines=(ExperimentConfig(),), *, max_items=6, per_line=2):
	items = []
	for field in dataclasses.fields(ExperimentConfig):
		name = field.name
		value = getattr(config, name)
		if any(value != getattr(baseline, name) for baseline in baselines):
			if name in CONFIG_LABELS:
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


@dataclasses.dataclass(frozen=True)
class RunUpdate:
	run_idx: int
	step: int
	loss: float | None
	img: Tensor | None = None


@dataclasses.dataclass(frozen=True)
class DoneUpdate:
	run_idx: int
	loss_history: list[float]


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
	parser.add_argument("--name", default=default, help="Display name for this run")


def make_config_parser(add_help=False):
	parser = argparse.ArgumentParser(add_help=add_help)
	add_config_args(parser, defaults=False)
	parser.add_argument("--workers", type=int, default=argparse.SUPPRESS, help="Maximum worker threads (default: one per run)")
	parser.add_argument("--output", "-o", default=argparse.SUPPRESS, help="Save the final visualization to this image file")
	return parser


def make_help_parser():
	parser = argparse.ArgumentParser(
		description="Attempt gradient inversion attacks, optionally running multiple concurrent experiments.",
		epilog=(
			"Options outside --run mutate the current baseline. --run creates an experiment from that "
			"baseline plus comma-separated overrides that do not affect later runs. Example: "
			"--seed 0 --lr 0.1 --opt Adam --run seed=1 --run seed=2 --opt SGD --run lr=0.01."
		)
	)
	add_config_args(parser, defaults=True)
	parser.set_defaults(**dataclasses.asdict(ExperimentConfig()), name=None)
	parser.add_argument("--workers", type=int, default=None, help="Maximum worker threads (default: one per run)")
	parser.add_argument("--output", "-o", default=None, help="Save the final visualization to this image file")
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
	experiments: list[ExperimentSpec] = []
	pending_tokens: list[str] = []

	def apply_pending():
		nonlocal config, next_name, workers, output, pending_tokens
		if not pending_tokens:
			return
		overrides = vars(parser.parse_args(expand_key_value_args(pending_tokens)))
		pending_tokens = []
		if "workers" in overrides:
			workers = overrides.pop("workers")
		if "output" in overrides:
			output = overrides.pop("output")
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
			if "workers" in run_overrides or "output" in run_overrides:
				parser.error("--workers and --output are global options and cannot be used inside --run overrides")
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
	return ParsedArgs(experiments, workers=max(1, workers), output=output)


def load_experiment_components(config: ExperimentConfig):
	img_split = config.img.split(':')
	if len(img_split) != 2 or img_split[0] not in DATASETS:
		raise ValueError(f"Invalid image spec {config.img!r}; expected dataset:idx")
	dataset = DATASETS[img_split[0]]
	inp, target = dataset[int(img_split[1])]
	if type(target) is int:
		target = one_hot(torch.tensor(target), len(dataset.classes)).to(dtype=torch.float32)
	model = MODELS[config.model](num_classes=len(dataset.classes), resolution=config.resolution)
	return model, inp, target


def run_experiment(spec: ExperimentSpec, updates: queue.Queue[RunUpdate | DoneUpdate | ErrorUpdate]):
	try:
		print(f"[{spec.name}] Running with config: {spec.config}")
		model, inp, target = load_experiment_components(spec.config)

		def plot_callback(step: int, img: Tensor | None, loss: float | None):
			plot_img = None if img is None else img.detach().cpu().clone()
			updates.put(RunUpdate(spec.idx, step, loss, plot_img))

		loss_history = gradinversion(
			model=model,
			inp=inp,
			truth_label=target,
			config=spec.config,
			plot_callback=plot_callback,
			run_name=spec.name
		)
		updates.put(DoneUpdate(spec.idx, loss_history))
	except Exception as e:
		updates.put(ErrorUpdate(spec.idx, e, traceback.format_exc()))


def make_target_image(spec: ExperimentSpec):
	model, inp, target = load_experiment_components(spec.config)
	truth_image, _ = prepare_truth_tensors(model, inp, target, spec.config.resolution, device="cpu")
	return truth_image


def process_plot_updates(updates: list[RunUpdate | DoneUpdate | ErrorUpdate], run_plots: dict[int, ImgPlot],
                         loss_histories: dict[int, list[float]], errors: list[ErrorUpdate],
                         grid: GridSpec, specs: list[ExperimentSpec]):
	loss_batches: dict[int, list[tuple[int, float]]] = {}
	image_updates: dict[int, RunUpdate] = {}

	for update in updates:
		match update:
			case RunUpdate(run_idx=run_idx, step=step, loss=loss, img=img):
				if loss is not None:
					loss_batches.setdefault(run_idx, []).append((step, loss))
				if img is not None:
					image_updates[run_idx] = update
			case DoneUpdate(run_idx=run_idx, loss_history=loss_history):
				loss_histories[run_idx] = loss_history
				print(f"[{specs[run_idx].name}] Finished after {len(loss_history)} optimization steps")
			case ErrorUpdate(run_idx=run_idx) as error:
				errors.append(error)
				print(f"[{specs[run_idx].name}] Failed: {error.error}")

	for spec in specs:
		plot = run_plots.get(spec.idx)
		img_update = image_updates.get(spec.idx)
		created_plot = False
		if plot is None and img_update is not None:
			plot = ImgPlot(
				grid[0, spec.idx + 1], img_update.img,
				title=spec.name, loss=img_update.loss,
				summary=format_config_summary(spec.config, baselines=(ExperimentConfig(), spec.baseline))
			)
			run_plots[spec.idx] = plot
			created_plot = True
		if plot is None:
			continue
		plot.add_losses(loss_batches.get(spec.idx, []))
		if img_update is not None and not created_plot:
			plot.add_img(img_update.img, img_update.step, img_update.loss)


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

	print(f"Starting {len(specs)} experiment(s) on {parsed.workers} worker thread(s)")
	with concurrent.futures.ThreadPoolExecutor(max_workers=parsed.workers, thread_name_prefix="gradinv") as executor:
		futures = [executor.submit(run_experiment, spec, updates) for spec in specs]

		while True:
			batch: list[RunUpdate | DoneUpdate | ErrorUpdate] = []
			try:
				batch.append(updates.get(timeout=0.05))
				while True:
					batch.append(updates.get_nowait())
			except queue.Empty:
				pass

			if batch:
				process_plot_updates(batch, run_plots, loss_histories, errors, grid, specs)
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
		fig.savefig(parsed.output, dpi=300)
		print(f"Saved visualization to {parsed.output}")
		sys.exit()


	plt.ioff()
	plt.show()


if __name__ == '__main__':
	main()
