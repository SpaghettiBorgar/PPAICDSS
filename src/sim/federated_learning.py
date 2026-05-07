import asyncio
import copy
import logging
from typing import List

import torch
from torchvision.transforms import InterpolationMode, v2

from sim import *
import xray_data
import xray_training
from sim.aggregator import Aggregator
from sim.fl_client import FederatedLearningClient
from sim.hospital import Hospital
from sim.transport import InProcessTransport
from util.weights import Weights
from util.random import random_partitions
from xray_cnn import XrayModel
from xray_data import XrayDataset

logger = logging.getLogger(__name__)


def create_participants(num_hospitals: int, params, cache_index=None, shm_manager=None, seed: int = 0) -> List[Hospital]:
	hospitals = []
	partitions = random_partitions(range(xray_data.TOTAL_SAMPLES), num_hospitals, seed=seed, evenness=0.8)
	for i, p in enumerate(partitions):
		hosp = Hospital(f"Hospital {i}")
		hosp.add_project('cxr', FederatedLearningClient(
			hosp.name,
			XrayDataset(offset=p.start, size=len(p), cache_index=cache_index, shm_manager=shm_manager, transform=params.transform),
		))
		hospitals.append(hosp)
	return hospitals


def initialize_participants(hospitals: List[Hospital], model: torch.nn.Module, weights: Weights, params, aggregator: Aggregator):
	for hosp in hospitals:
		client = hosp.fl_projects['cxr']
		client.set_model(copy.deepcopy(model).to(params.device))
		client.set_weights(weights.detach())
		client.set_params(params)
		establish_connection(client, aggregator)
		client.start()


def setup_federation(hospitals: List[Hospital]):
	join_tasks = [asyncio.create_task(hosp.fl_projects['cxr'].join_federation()) for hosp in hospitals]
	return asyncio.gather(*join_tasks)


def establish_connection(client: FederatedLearningClient, aggregator: Aggregator):
	transport = InProcessTransport(latency=0.5)
	client.connect(transport.create_socket())
	aggregator.connect(transport.create_socket())


async def run_simulation():
	device = "cuda"

	init_model = XrayModel()
	init_weights = Weights(init_model)
	class_weights = xray_data.TOTAL_SAMPLES / torch.tensor(list(xray_data.CLASS_WEIGHTS.values())) - 1
	class_weights = class_weights.to(device=device)

	params = xray_training.make_params('testing', device=device)
	params.transform = v2.Compose([
		v2.Resize(size=None, max_size=params.resolution, interpolation=InterpolationMode.BICUBIC),
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		# v2.CenterCrop([params.resolution, params.resolution])
	])
	params.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
	params.optimizer = torch.optim.AdamW(init_model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

	hospitals = create_participants(3, params)
	aggregator = Aggregator()
	aggregator.start()
	logger.info("Aggregator started")
	initialize_participants(hospitals, init_model, init_weights, params, aggregator)
	logger.info("Clients initialized")
	await setup_federation(hospitals)
	logger.info("Federation set up")

	sim_start_time = time.time()
	await aggregator.start_new_round()
	await aggregator.loop_task


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)
	asyncio.run(run_simulation(), debug=True)
