from types import SimpleNamespace

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

import training
import xray_data
from xray_cnn import *

phases = {
	"testing": dict(
		batchsize=8,
		batches=10,
		epochs=3,
		resolution=224,
		lr=1e-3,
		weight_decay=1e-3,
		freeze_backend=True,
		shuffle=False
	),
	1: dict(
		batchsize=512,
		batches=80,
		epochs=3,
		resolution=384,
		lr=1e-3,
		weight_decay=1e-3,
		freeze_backend=True,
		shuffle=False
	),
	2: dict(
		batchsize=128,
		batches=256,
		epochs=10,
		resolution=384,
		lr=1e-4,
		weight_decay=1e-3,
		freeze_backend=False,
		shuffle=False
	),
	3: dict(
		batchsize=64,
		batches=64,
		epochs=4,
		resolution=600,
		lr=1e-4,
		weight_decay=1e-4,
		freeze_backend=False,
		shuffle=False
	)
}
params = None


def train_xray_model(phase, checkpoint, device):
	global params
	params = SimpleNamespace(**phases[phase])
	vars(params).update(
		phase=phase,
		checkpoint=checkpoint,
		device=device
	)

	weights = None if params.checkpoint is None else torch.load(f"{checkpoints_dir}/{params.checkpoint}", weights_only=True)
	model = XrayModel(num_labels=14, xray_view_dim=5, weights=weights).to(device)

	class_weights = xray_data.total_samples / torch.tensor(list(xray_data.class_weights.values())) - 1
	class_weights = class_weights.to(device=device)

	criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
	transform = v2.Compose([
		v2.Resize(size=None, max_size=params.resolution, interpolation=InterpolationMode.BICUBIC),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[(0.485 + 0.456 + 0.406) / 3], std=[(0.229 + 0.224 + 0.225) / 3]),
		v2.CenterCrop([params.resolution, params.resolution])
	]) # TODO

	vars(params).update(
		criterion=criterion,
		optimizer=optimizer
	)

	if params.freeze_backend:
		for param in model.densenet.parameters():
			param.requires_grad = False

	# num_workers = min(os.cpu_count(), 16)
	num_workers=2
	training_data = xray_data.XrayDataset(img_root, offset=0, size=200000, transform=transform)
	testing_data = xray_data.XrayDataset(img_root, offset=-20, size=0, transform=transform)
	train_dataloader = DataLoader(training_data, batch_size=params.batchsize, num_workers=num_workers, pin_memory=True, shuffle=params.shuffle)
	test_dataloader = DataLoader(testing_data, batch_size=params.batchsize, num_workers=num_workers, pin_memory=True, shuffle=params.shuffle)

	training.run(model, params, train_dataloader, test_dataloader)
