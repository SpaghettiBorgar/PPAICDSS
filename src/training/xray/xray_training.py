from torch.utils.data import DataLoader, Dataset

import data_prep
import training.xray.xray_data as xray_data
from training import training
from training.params import Params
from training.xray.xray_cnn import *
from training.xray.xray_params import XrayParams
from util.sampler import BlockShuffleBatchSampler


def register_faulthandler(*args):
	__import__('faulthandler').enable()


def make_loader_args(num_workers: int = 0, **kwargs):
	return dict(
		pin_memory=True,
		num_workers=num_workers,
		prefetch_factor=4 if num_workers > 0 else None,
		persistent_workers=num_workers > 0,
		worker_init_fn=register_faulthandler if num_workers > 0 else None,
		**kwargs
	)


def make_train_loader(params: Params, dataset: Dataset, **kwargs):
	loader_args = make_loader_args(**kwargs)
	train_loader_args = (
		dict(batch_sampler=params.sampler) if isinstance(params.sampler, torch.utils.data.Sampler) else
		dict(shuffle=True) if params.sampler == "shuffle" else
		dict(batch_size=params.batch_size) if params.sampler == "default" else dict()
	)
	return DataLoader(dataset=dataset, **(loader_args | train_loader_args))


def make_test_loader(params: Params, dataset: Dataset, **kwargs):
	loader_args = make_loader_args(**kwargs)
	test_loader_args = dict(batch_size=params.batch_size)
	return DataLoader(dataset=dataset, **(loader_args | test_loader_args))


def train_xray_model(phase, checkpoint, device):
	params = XrayParams(phase, checkpoint=checkpoint, device=device)

	model = params.get_model()
	transform = params.get_transform()

	xray_data.setup_shm()

	training_data = xray_data.XrayDataset(img_root, offset=0, size=200000, transform=transform)
	testing_data = xray_data.XrayDataset(img_root, offset=-4000, size=0, transform=transform)

	params.sampler = BlockShuffleBatchSampler(
		len(training_data), data_prep.chunk_size, block_size=data_prep.chunk_size // 4, batch_size=params.batch_size)

	loader_args = dict(
		num_workers=min(os.cpu_count() or 0, 8)
	)

	train_dataloader = make_train_loader(params, training_data, **loader_args)
	test_dataloader = make_test_loader(params, testing_data, **loader_args)

	try:
		training.run(model, params, train_dataloader, test_dataloader)
	finally:
		xray_data.shutdown_shm()
