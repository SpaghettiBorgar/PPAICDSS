import sys
import asyncio
import logging
from typing import List

import torch

import sim.time
import sim.gvars as gvars
from sim.aggregator import Aggregator
from sim.fl_client import FederatedLearningClient
from sim.hospital import Hospital
from sim.transport import InProcessTransport, inprocess_address_space
from training import training
from training.xray import xray_data, xray_training
from training.xray.xray_data import XrayDataset
from training.xray.xray_params import XrayParams, PHASES as XRAY_PHASES
from util.utils import random_partitions, auto_type
import warnings
import re
warnings.filterwarnings("ignore", category=UserWarning, message=r"Full backward hook is firing .*")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Secure RNG turned off.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Optimal order is the.*")

logger = logging.getLogger(__name__)

logging.getLogger("opacus.validators.batch_norm").setLevel("WARNING")
logging.getLogger("opacus.validators.module_validator").setLevel("WARNING")

def create_participants(num_hospitals: int, seed: int = None, devices=None) -> List[Hospital]:
	hospitals = []
	partitions = random_partitions(range(xray_data.TRAIN_SIZE), num_hospitals, seed=seed, evenness=0.8)
	for i, p in enumerate(partitions):
		hosp = Hospital(f"Hospital {i}", device=devices[i % len(devices)] if devices is not None else "cuda")
		hosp.add_project('cxr', FederatedLearningClient(
			hosp.name,
			XrayDataset(offset=p.start, size=len(p)),
		))
		hospitals.append(hosp)
		inprocess_address_space[hosp.name] = hosp.fl_projects['cxr']
	return hospitals


def initialize_participants(hospitals: List[Hospital], aggregator: Aggregator, phase='testing', **kwargs):
	for hosp in hospitals:
		params = XrayParams(phase, device=hosp.device, **kwargs)
		client = hosp.fl_projects['cxr']
		client.set_params(params, init=True)
		client.dataset.transform = params.get_transform()
		establish_connection(client, aggregator)
		client.start()

def update_params(hospitals: List[Hospital], phase, **kwargs):
	for hosp in hospitals:
		params = XrayParams(phase, device=hosp.device, **kwargs)
		flp = hosp.fl_projects['cxr']
		params.privacy_engine = flp.training_params.privacy_engine
		params._model = flp.model
		params._optimizer = flp.training_params._optimizer
		params._criterion = flp.training_params._criterion
		flp.trained_epochs = 0
		flp.set_params(params, init=False)

def setup_federation(hospitals: List[Hospital]):
	join_tasks = [asyncio.create_task(hosp.fl_projects['cxr'].join_federation()) for hosp in hospitals]
	return asyncio.gather(*join_tasks)


def establish_connection(client: FederatedLearningClient, aggregator: Aggregator):
	transport = InProcessTransport(latency=0.5)
	client.connect(transport.create_socket(), 'aggregator')
	aggregator.connect(transport.create_socket())


async def run_simulation(num_participants=3, phases=['testing'], checkpoint=None, devices=None, seed=None, **extra_params):
	device_count = torch.cuda.device_count()
	devices = [f"cuda:{i}" for i in range(device_count)] if devices is None else devices

	xray_data.setup_shm()

	hospitals = create_participants(num_participants, devices=devices, seed=seed)
	aggregator = Aggregator()
	inprocess_address_space['aggregator'] = aggregator
	
	for phase_i, phase in enumerate(phases):
		params = dict(checkpoint=checkpoint, save=False) | extra_params

		epochs_remaining = XRAY_PHASES[phase]['epochs']

		logger.info(f"\n\n\nStarting phase {phase} with {epochs_remaining} epochs\n\n\n")

		if phase_i == 0:
			aggregator.start()
			logger.info("Aggregator started")
			initialize_participants(hospitals, aggregator, phase=phase, **params)
			logger.info("Clients initialized")
			await setup_federation(hospitals)
			logger.info("Federation set up")
		else:
			update_params(hospitals, phase, **params)
			logger.info("Updated client params")

		logs = dict(
			time=[],
			loss=[],
			acc=[]
		)

		test_params = XrayParams(**(params | dict(resolution=600, batch_size=256, device="cuda")))
		test_dataset = XrayDataset(offset=xray_data.TEST_OFFSET, transform=test_params.get_transform())
		test_loader = xray_training.make_test_loader(test_params, test_dataset)

		sim.time.init_time()

		rounds = (epochs_remaining + gvars.fl_params.epochs_per_round - 1) // gvars.fl_params.epochs_per_round
		for round in range(rounds):
			logger.info("Starting round %d", round)

			await aggregator.start_new_round()
			await aggregator.round_end_event.wait()

			models = [hosp.fl_projects['cxr'].model for hosp in hospitals]
			logs['time'].append([hosp.fl_projects['cxr'].training_times for hosp in hospitals])
			logs['loss'].append([hosp.fl_projects['cxr'].training_losses for hosp in hospitals])
			logs['acc'].append(training.test(models[0], test_params, test_loader))

			save_params = hospitals[0].fl_projects['cxr'].training_params.__dict__.copy()

		logger.info(f"Phase {phase} done")
		models = [hosp.fl_projects['cxr'].model for hosp in hospitals]

		save_params['phase'] = f"fl_{phase}"
		save_params['argv'] = sys.argv
		training.save(models[0], params, logs)

	logger.info("Rounds ended, shutting down")
	aggregator.shutdown()
	for hosp in hospitals:
		hosp.fl_projects['cxr'].shutdown()

	logger.info("Simulation done")


def parse_args():
	import argparse
	parser = argparse.ArgumentParser(description="Run Federated Learning Simulation")
	parser.add_argument("--num-participants", "-n", type=int, default=3, help="Number of participants in the simulation")
	parser.add_argument("--epochs-per-round", "-r", type=int, default=2, help="Number of epochs per round")
	parser.add_argument("--phase", "-p", action='append', type=str, default=[], help="Training phases for parameter presets")
	parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint to load")
	parser.add_argument("--devices", "-d", action='append', type=str, default=None, help="Devices to use for participants (default: all available GPUs)")
	parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
	parser.add_argument("--use-smpc", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=True, help="Use SMPC for secure aggregation")
	parser.add_argument("--round-timeout", type=float, default=float('inf'), help="Time limit for each round")
	parser.add_argument("--debug", action='store_true', help="Enable debug logging")
	parser.add_argument("-P", action='append', default=[], help="Override training parameters")
	return parser.parse_args()

def main():
	args = parse_args()
	gvars.fl_params.use_smpc = args.use_smpc
	gvars.epochs_per_round = args.epochs_per_round
	gvars.round_timeout = args.round_timeout
	
	extra_params = {k: auto_type(v) for (k, v) in [p.partition('=')[::2] for p in args.P]}

	assert (('target_epsilon' in extra_params) == ('target_delta' in extra_params) == ('grad_norm' in extra_params)
		) or (('grad_norm' in extra_params) == ('noise_mult' in extra_params))

	logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, force=True)
	logger.info(f"Starting simulation with arguments: {args}\nGlobal vars:{gvars.fl_params}")

	asyncio.run(run_simulation(
		num_participants=args.num_participants,
		phases=args.phase,
		checkpoint=args.checkpoint,
		devices=args.devices,
		seed=args.seed,
		**extra_params
	), debug=args.debug)

if __name__ == '__main__':

	main()

	logger.info("Terminating")
	sys.exit(0)
