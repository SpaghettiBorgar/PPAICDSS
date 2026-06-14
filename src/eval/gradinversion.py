import argparse
import functools
import inspect
import os
import signal
import sys
import time
from typing import List, Callable, Any

import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.image import AxesImage
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.utils import make_grid

from models.simplenets import SimpleNet, LeNet
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


OPTIMIZERS = {k: getattr(torch.optim, k) for k in torch.optim.__all__}
MODELS = {
			 cls.__name__: cls for cls in
			 [SimpleNet, LeNet, XrayModel]
		 } | {
			 name: functools.partial(resnet_wrapper, block, layers, weights=None)
			 for name, layers, block, weights in [
		('ResNet0', [0, 0, 0, 0], BasicBlock, None),
		('ResNet18', [2, 2, 2, 2], BasicBlock, torchvision.models.ResNet18_Weights.DEFAULT),
		('ResNet34', [3, 4, 6, 3], BasicBlock, torchvision.models.ResNet34_Weights.DEFAULT),
		('ResNet50', [3, 4, 6, 3], Bottleneck, torchvision.models.ResNet50_Weights.DEFAULT)
	]}
DATASETS = {
	'CXR': Lazy(XrayDataset, use_chunks=False),
	'CIFAR10': Lazy(torchvision.datasets.CIFAR10, root=f"{DATA_DIR}/torch_datasets", download=True),
	"MNIST": Lazy(torchvision.datasets.MNIST, root=f"{DATA_DIR}/torch_datasets", download=True)
}


def _process_img(img: Tensor):
	return make_grid(img.detach(), nrow=1, normalize=True).permute(1, 2, 0).cpu().clone()


class ImgPlot:
	gs: SubplotSpec
	subgs: GridSpec
	img_history: List[tuple[int, Tensor]]
	ax_img: Axes
	ax_slider: Axes
	img: AxesImage
	slider: Slider

	def __init__(self, gs: SubplotSpec, init_img: Tensor):
		self.gs = gs
		self.subgs = gs.subgridspec(2, 1, height_ratios=[10, 1])

		self.img_history = [(0, _process_img(init_img))]

		self.ax_img = plt.gcf().add_subplot(self.subgs[0])
		self.img = self.ax_img.imshow(self.img_history[0][1])

		self.ax_slider = plt.gcf().add_subplot(self.subgs[1])

		self.slider = Slider(self.ax_slider, 'Step', valmin=0, valmax=1, valinit=1, valstep=1)
		self.slider.on_changed(lambda val: self.update_plot())

	def update_plot(self):
		idx = min(int(self.slider.val), len(self.img_history) - 1)
		iter_num, img = self.img_history[idx]
		self.img.set_data(img)
		self.ax_img.set_title(f"Recovered Image (i={iter_num})")
		redraw_plot()

	def add_img(self, img, idx):
		self.img_history.append((idx, _process_img(img)))
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

	def hide_slider(self):
		self.slider.set_active(False)
		self.slider.label.set_visible(False)
		rect = plt.Rectangle((0, 0), 1, 1, transform=self.ax_slider.transAxes,
		                     color='white', zorder=10)
		self.ax_slider.add_patch(rect)


fig = None


def redraw_plot():
	fig.canvas.draw()
	fig.canvas.flush_events()


stop = False


def handle_int(*args):
	global stop
	if stop:
		sys.exit()
	else:
		stop = True
		print("Stopping, press again to exit")


def recover_image(model: nn.Module, gradients: List[Tensor], img: Tensor, labels: Tensor, criterion: nn.Module, optimizer: str, lr: float,
                  *, max_iterations=3000, timeout=15 * 60, noise_mixin=0., iter_callback: Callable[[int, Tensor, float], Any] | None = None):
	optimizer = OPTIMIZERS[optimizer]([img, labels], lr=lr)
	img.requires_grad = True
	labels.requires_grad = True

	i = 1
	start_time = time.time()
	while not (stop or (max_iterations and i >= max_iterations) or (timeout and time.time() - start_time > timeout)):
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
			img += torch.randn_like(img) * noise_mixin

		if torch.isnan(img).any():
			print("NAN")
			raise ValueError()

		if iter_callback is not None:
			iter_callback(i, img, loss_val)

		if i < 10 or i % 5 == 0:
			print(f"Iteration {i}: Gradient Loss = {loss_val:.5f}, diff = {diff_img.min():.3E}/{diff_img.max():.3E}/{diff_img.std():.3E}")


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
		if isinstance(model, SimpleNet):
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


def gradinversion(model: nn.Module, inp, truth_label: Tensor, optimizer: str = "LBFGS", lr=0.01, resolution=64, device="cpu",
                  target_mixin=0., noise_mixin=0., max_iterations=3000, timeout=15 * 60, seed=0, gs=None, gs_target=None, **_):
	torch.manual_seed(seed)
	model = model.to(device)
	# criterion = nn.BCEWithLogitsLoss(pos_weight=xray_params.CLASS_WEIGHTS.to(device))
	criterion = nn.BCEWithLogitsLoss()

	transform = v2.Compose([
		v2.Resize(size=None, max_size=resolution, interpolation=InterpolationMode.BICUBIC),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.CenterCrop([resolution, resolution]),
		torch.nan_to_num
	])
	truth_image = adapt_input_to_model(inp, model, transform)
	truth_label = truth_label
	truth_image, truth_label = tree_map(lambda t: t.unsqueeze(0).to(device), (truth_image, truth_label))

	if gs_target is not None:
		ImgPlot(gs_target, truth_image).hide_slider()

	recovered_image = torch.rand_like(truth_image).to(device)
	print(truth_image.min(), truth_image.max(), recovered_image.min(), recovered_image.max())
	recovered_image = (1. - target_mixin) * recovered_image + target_mixin * truth_image

	recovered_label_logits = torch.randn_like(truth_label).to(device)

	print(f"Total model parameters: {sum([torch.numel(p) for p in model.parameters()])}")
	print(f"Total gradinversion parameters: {sum([torch.numel(recovered_image), torch.numel(recovered_label_logits)])}")

	plot = None if gs is None else ImgPlot(gs, recovered_image)

	model.eval()
	model.zero_grad()

	pred = model(truth_image)
	loss = criterion(pred, truth_label)
	print(f"real loss: {loss}")

	real_grads = torch.autograd.grad(loss, model.parameters())
	real_grads = [g.detach().clone() for g in real_grads]

	loss_history = []

	def callback(i, img, loss):
		loss_history.append(loss)
		if plot is not None:
			if i < 10 or i % 10 == 0:
				plot.add_img(img, i)
			redraw_plot()

	recover_image(model, real_grads, recovered_image, recovered_label_logits, criterion, optimizer, lr, max_iterations=max_iterations, timeout=timeout, noise_mixin=noise_mixin, iter_callback=callback)

	return loss_history


def load_sample(path):
	splits = path.split(':')
	return DATASETS[splits[0]][splits[1]]


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", help="Device to use")
	parser.add_argument("--max-iterations", "-I", type=int, help="Maximum iterations (0 for unlimited)")
	parser.add_argument("--timeout", type=int, help="Maxiumum time (s)")
	parser.add_argument("--optimizer", "--opt", choices=OPTIMIZERS)
	parser.add_argument("--lr", "--learning-rate", type=float)
	parser.add_argument("--seed", type=int)
	parser.add_argument("--resolution", "-R", type=int)
	parser.add_argument("--model", choices=MODELS, default='LeNet')
	parser.add_argument("--img", default="CXR:0", help=f"Input sample to recreate (dataset:idx)\nAvailable datasets: {{{', '.join(DATASETS.keys())}}}")
	parser.add_argument("--target-mixin", type=float, help="Percentage of target to mix into starting image")
	parser.add_argument("--noise-mixin", "-N", type=float, help="Noise amplitude to mix in each iteration")

	parser.set_defaults(**{
		k: v.default
		for k, v in inspect.signature(gradinversion).parameters.items() if v.default is not inspect._empty
	})

	return parser.parse_args()


def main():
	args = parse_args()

	img_split = args.img.split(':')
	dataset = DATASETS[img_split[0]]
	inp, target = dataset[int(img_split[1])]
	if type(target) is int:
		target = one_hot(torch.tensor(target), len(dataset.classes)).to(dtype=torch.float32)
	model = MODELS[args.model](num_classes=len(dataset.classes), resolution=args.resolution)

	signal.signal(signal.SIGINT, handle_int)

	plt.ion()
	global fig
	fig = plt.figure()
	grid = fig.add_gridspec(1, 2)
	gs0 = grid[0, 0]
	gs1 = grid[0, 1]

	loss_history = gradinversion(**(args.__dict__ | dict(gs_target=gs0, gs=gs1, model=model, inp=inp, truth_label=target)))

	plt.ioff()
	plt.figure(figsize=(6, 4))
	plt.plot(loss_history)
	plt.xlabel("Iteration")
	plt.ylabel("Gradient Matching Loss")
	plt.title("Attack Optimization Progress")
	plt.show()


if __name__ == '__main__':
	main()
