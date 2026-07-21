import inspect
from functools import wraps

import torch
from PIL import Image
from torchvision.transforms import v2, InterpolationMode

from training.xray.xray_data import LABELS as XRAY_LABELS
from training.xray.xray_params import XrayParams


def get_weather(location: str) -> str:
	"""Get the weather at a location.

	Args:
		location: The city name, e.g. San Francisco
	"""

	return f"It's sunny in {location}."


def analyze_xray(img: Image.Image) -> dict[str, float]:
	"""
	Analyze an X-Ray image for likelihood of several conditions

	Args:
		img: The image to analyze
	"""

	params = XrayParams(checkpoint="latest")
	model = params.get_model()
	transform = v2.Compose([
		v2.PILToTensor(),
		v2.ToImage(),
		v2.Resize(size=None, max_size=512, interpolation=InterpolationMode.BICUBIC),
		v2.ToDtype(torch.float32, scale=True),
	])
	inp = transform(img).unsqueeze(0).to(params.device)

	model.eval()
	output = model(inp).squeeze().sigmoid()

	values = [round(v, 4) for v in output.tolist()]
	results = dict(zip(XRAY_LABELS, values))

	return results


class ImageContext:
	def __init__(self, images=None):
		self.images = images if images is not None else []

	def add(self, *images):
		self.images.extend(images)

	def get(self, idx):
		return self.images[int(idx)]

	def bind(self, func):
		sig = inspect.signature(func)

		image_params = {
			name
			for name, param in sig.parameters.items()
			if param.annotation is Image.Image
		}

		@wraps(func)
		def wrapper(*args, **kwargs):
			bound = sig.bind(*args, **kwargs)

			for name in image_params:
				bound.arguments[name] = self.get(bound.arguments[name])

			return func(*bound.args, **bound.kwargs)

		new_params = [p.replace(annotation=int) for p in sig.parameters.values() if p.annotation is Image.Image]

		wrapper.__signature__ = sig.replace(parameters=new_params)

		return wrapper
