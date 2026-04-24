import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, v2

import data_prep
import sampler
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
		save=False
	),
	1: dict(
		batchsize=512,
		epochs=6,
		resolution=384,
		lr=1e-3,
		weight_decay=1e-3,
		freeze_backend=True,
	),
	2: dict(
		batchsize=160,
		epochs=10,
		resolution=384,
		lr=1e-4,
		weight_decay=1e-3,
		freeze_backend=False,
	),
	3: dict(
		batchsize=64,
		epochs=4,
		resolution=600,
		lr=1e-4,
		weight_decay=1e-4,
		freeze_backend=False,
	)
}
params = None

default_params = dict(
	batches=0,
	shuffle=True,
	freeze_backend=False,
	save=True
)


def register_faulthandler(*args):
	__import__('faulthandler').enable()


def train_xray_model(phase, checkpoint, device):
	global params
	params = default_params | phases[phase]
	params = SimpleNamespace(**params)
	vars(params).update(
		phase=phase,
		checkpoint=checkpoint,
		device=device
	)

	weights = None if params.checkpoint is None else torch.load(f"{checkpoints_dir}/{params.checkpoint}", weights_only=True)
	model = XrayModel(num_labels=14, xray_view_dim=5, weights=weights).to(device)

	class_weights = xray_data.TOTAL_SAMPLES / torch.tensor(list(xray_data.CLASS_WEIGHTS.values())) - 1
	class_weights = class_weights.to(device=device)

	criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
	optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
	transform = v2.Compose([
		v2.Resize(size=None, max_size=params.resolution, interpolation=InterpolationMode.BICUBIC),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		# v2.CenterCrop([params.resolution, params.resolution])
	])

	vars(params).update(
		criterion=criterion,
		optimizer=optimizer
	)

	if params.freeze_backend:
		for param in model.densenet.parameters():
			param.requires_grad = False

	num_workers = min(os.cpu_count(), 8)

	manager = mp.Manager()
	smm = SharedMemoryManager()
	smm.start()
	shared_index = manager.dict()
	shared_index['lock'] = manager.Lock()

	training_data = xray_data.XrayDataset(img_root, cache_index=shared_index, shm_manager=smm, offset=0, size=200000, transform=transform)
	testing_data = xray_data.XrayDataset(img_root, cache_index=shared_index, shm_manager=smm, offset=-4000, size=0, transform=transform)

	loader_args = dict(
		pin_memory=True,
		num_workers=num_workers,
		prefetch_factor=4 if num_workers > 0 else None,
		persistent_workers=num_workers > 0,
		worker_init_fn=register_faulthandler if num_workers > 0 else None
	)

	train_dataloader = DataLoader(training_data, **loader_args, **(
		dict(batch_sampler=sampler.BlockShuffleBatchSampler(len(training_data), data_prep.chunk_size, block_size=data_prep.chunk_size // 4, batch_size=params.batchsize))
		if params.shuffle else
		dict(batch_size=params.batchsize)))
	test_dataloader = DataLoader(testing_data, **loader_args, batch_size=params.batchsize)

	try:
		training.run(model, params, train_dataloader, test_dataloader)
	finally:
		smm.shutdown()
		manager.shutdown()
