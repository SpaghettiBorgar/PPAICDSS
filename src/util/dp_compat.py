import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from training.params import Params
import collections
from util.weights import Weights

def _safe_basic_block_forward(self, x):
	identity = x

	out = self.conv1(x)
	out = self.bn1(out)
	out = self.relu(out)

	out = self.conv2(out)
	out = self.bn2(out)

	if self.downsample is not None:
		identity = self.downsample(x)

	out = out + identity
	out = self.relu(out)

	return out


def _safe_bottleneck_forward(self, x):
	identity = x

	out = self.conv1(x)
	out = self.bn1(out)
	out = self.relu(out)

	out = self.conv2(out)
	out = self.bn2(out)
	out = self.relu(out)

	out = self.conv3(out)
	out = self.bn3(out)

	if self.downsample is not None:
		identity = self.downsample(x)

	out = out + identity
	out = self.relu(out)

	return out


def _safe_densenet_forward(self, x: torch.Tensor) -> torch.Tensor:
	features = self.features(x)
	out = F.relu(features, inplace=False)
	out = F.adaptive_avg_pool2d(out, (1, 1))
	out = torch.flatten(out, 1)
	out = self.classifier(out)
	return out


def make_model_dp_compatible(model: nn.Module, swap_blocks = True) -> nn.Module:
	for module in model.modules():
		if isinstance(getattr(module, "inplace", None), bool):
			module.inplace = False
		if swap_blocks:
			if isinstance(module, BasicBlock):
				module.forward = types.MethodType(_safe_basic_block_forward, module)
			elif isinstance(module, Bottleneck):
				module.forward = types.MethodType(_safe_bottleneck_forward, module)
			elif isinstance(module, DenseNet):
				module.forward = types.MethodType(_safe_densenet_forward, module)
	return model

import operator
import torch.fx as fx

# https://github.com/meta-pytorch/opacus/issues/828

_INPLACE_OPS = {
	operator.iadd: operator.add,
	operator.isub: operator.sub,
	operator.imul: operator.mul,
	operator.itruediv: operator.truediv,
}

def fix_inplace(model: nn.Module) -> nn.Module:
	"""Replace all in-place operations in ``model`` with out-of-place equivalents.

	Flips module-level ``inplace`` flags (e.g. ``ReLU``), then uses
	``torch.fx.symbolic_trace`` to rewrite ``+=`` and in-place methods/functions.
	"""

	for m in model.modules():
		if hasattr(m, "inplace"):
			m.inplace = False

	gm = fx.symbolic_trace(model)

	for node in gm.graph.nodes:
		if node.op == "call_function":
			if node.target in _INPLACE_OPS:
				node.target = _INPLACE_OPS[node.target]
			elif callable(node.target):
				name = getattr(node.target, "__name__", "")
				if name.endswith("_") and not name.endswith("__"):
					oop = getattr(torch, name[:-1], None)
					if oop is not None:
						node.target = oop

		elif node.op == "call_method" and isinstance(node.target, str):
			if node.target.endswith("_") and not node.target.endswith("__"):
				stripped = node.target[:-1]
				if hasattr(torch.Tensor, stripped):
					node.target = stripped

	gm.graph.lint()
	gm.recompile()
	return gm

def check_dp_params(params: dict | Params):
	params = params.__dict__

	if not ((('target_epsilon' in params) == ('target_delta' in params) == ('grad_norm' in params)
		) or (('grad_norm' in params) == ('noise_mult' in params)) and (('noise_mult' in params) != 'target_epsilon' in params)):
		raise ValueError("Invalid DP parameters: must specify either (target_epsilon, target_delta, grad_norm) or (grad_norm, noise_mult), but not both.")
	return 'grad_norm' in params

def make_private_auto(model, optimizer, data_loader, params):
	from util.utils import fix_collate
	fix_collate(data_loader)
	check_dp_params(params)
	if params.target_epsilon is not None and params.target_delta is not None and params.grad_norm is not None:
		print(f"Applying DP with target epsilon {params.target_epsilon} and delta {params.target_delta}")
		model, optimizer, data_loader = params.privacy_engine.make_private_with_epsilon(
			module=model, optimizer=optimizer, data_loader=data_loader, target_delta=params.target_delta, target_epsilon=params.target_epsilon, epochs=params.epochs, max_grad_norm=params.grad_norm)
	elif params.grad_norm is not None and params.noise_mult is not None:
		print(f"Applying DP with noise multiplier {params.noise_mult} and grad norm {params.grad_norm}")
		model, optimizer, data_loader = params.privacy_engine.make_private(
			module=model, optimizer=optimizer, data_loader=data_loader, noise_multiplier=params.noise_mult, max_grad_norm=params.grad_norm)
	else:
		print("DP not enabled")
	
	return model, optimizer, data_loader

def convert_dp_state_dict(weights: collections.OrderedDict):
	return collections.OrderedDict((n.removeprefix("_module."), p) for n, p in weights.items())