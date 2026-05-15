import asyncio
import logging
from typing import List

import torch

import sim.time
from sim.aggregator import Aggregator
from sim.fl_client import FederatedLearningClient
from sim.hospital import Hospital
from sim.transport import InProcessTransport
from training import training
from training.xray import xray_data, xray_training
from training.xray.xray_data import XrayDataset
from training.xray.xray_params import XrayParams
from util.random import random_partitions
from util.weights import Weights

logger = logging.getLogger(__name__)


def create_participants(num_hospitals: int, seed: int = 0, devices=None) -> List[Hospital]:
	hospitals = []
	partitions = random_partitions(range(xray_data.TOTAL_SAMPLES), num_hospitals, seed=seed, evenness=0.8)
	for i, p in enumerate(partitions):
		hosp = Hospital(f"Hospital {i}", device=devices[i % len(devices)] if devices is not None else "cuda")
		hosp.add_project('cxr', FederatedLearningClient(
			hosp.name,
			XrayDataset(offset=p.start, size=len(p)),
		))
		hospitals.append(hosp)
	return hospitals


def initialize_participants(hospitals: List[Hospital], aggregator: Aggregator, **kwargs):
	for hosp in hospitals:
		params = XrayParams('testing', device=hosp.device, **kwargs)
		client = hosp.fl_projects['cxr']
		client.set_params(params)
		client.dataset.transform = params.get_transform()
		client.set_model(params.get_model())
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
	device_count = torch.cuda.device_count()
	devices = [f"cuda:{i}" for i in range(device_count)]

	xray_data.setup_shm()

	hospitals = create_participants(3, devices=devices)
	aggregator = Aggregator()
	aggregator.start()
	logger.info("Aggregator started")
	initialize_participants(hospitals, aggregator)
	logger.info("Clients initialized")
	await setup_federation(hospitals)
	logger.info("Federation set up")

	await aggregator.start_new_round()
	logger.info("Waiting for round end")
	await aggregator.round_end_event.wait()
	sim.time.init_time()
	logger.info("Rounds ended, shutting down")
	aggregator.shutdown()
	for hosp in hospitals:
		hosp.fl_projects['cxr'].shutdown()
	logger.info("Simulation done")

	xray_data.shutdown_shm()

if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	asyncio.run(run_simulation(), debug=True)
	logger.info("Terminating")
