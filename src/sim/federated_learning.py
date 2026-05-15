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


def initialize_participants(hospitals: List[Hospital], aggregator: Aggregator, phase='testing', **kwargs):
	for hosp in hospitals:
		params = XrayParams(phase, device=hosp.device, **kwargs)
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


async def run_simulation(num_participants=3, phase='testing', rounds=3, checkpoint=None, **extra_params):
	device_count = torch.cuda.device_count()
	devices = [f"cuda:{i}" for i in range(device_count)]

	xray_data.setup_shm()

	params = dict(checkpoint=checkpoint, phase=phase, save=False) | extra_params

	hospitals = create_participants(num_participants, devices=devices)
	aggregator = Aggregator()
	aggregator.start()
	logger.info("Aggregator started")
	initialize_participants(hospitals, aggregator, **params)
	logger.info("Clients initialized")
	await setup_federation(hospitals)
	logger.info("Federation set up")

	logs = dict(
		time=[],
		loss=[],
		acc=[]
	)

	test_params = XrayParams(**(params | dict(resolution=600, batch_size=256, device="cuda")))
	test_dataset = XrayDataset(offset=-20000, transform=test_params.get_transform())
	test_loader = xray_training.make_test_loader(test_params, test_dataset)

	sim.time.init_time()

	for round in range(rounds):
		logger.info("Starting round %d", round)

		await aggregator.start_new_round()
		logger.info("Waiting for round end")
		await aggregator.round_end_event.wait()

		models = [hosp.fl_projects['cxr'].model for hosp in hospitals]
		logs['time'].append([hosp.fl_projects['cxr'].training_times for hosp in hospitals])
		logs['loss'].append([hosp.fl_projects['cxr'].training_losses for hosp in hospitals])
		logs['acc'].append(training.test(models[0], test_params, test_loader))

		params = hospitals[0].fl_projects['cxr'].training_params

	models = [hosp.fl_projects['cxr'].model for hosp in hospitals]
	with torch.no_grad():
		for (n, p) in iter(Weights(models[1]) - Weights(models[0])):
			print(f"{n}: {(p.min().item(), p.max().item(), p.mean().item(), p.std().item())}")
		print()
		for (n, p) in iter(Weights(models[2]) - Weights(models[0])):
			print(f"{n}: {(p.min().item(), p.max().item(), p.mean().item(), p.std().item())}")

	training.save(models[0], params, logs, path_fmt="./checkpoints/xray_resnet/fl_%s")

	logger.info("Rounds ended, shutting down")
	aggregator.shutdown()
	for hosp in hospitals:
		hosp.fl_projects['cxr'].shutdown()

	logger.info("Simulation done")


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG)
	asyncio.run(run_simulation(3, 'testing', 2, None, epochs=2, batch_size=256, batches=256), debug=True)
	logger.info("Terminating")
