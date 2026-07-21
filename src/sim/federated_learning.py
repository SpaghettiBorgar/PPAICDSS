import asyncio
import faulthandler
import logging
import os
import signal
import sys
import warnings
from typing import List

import torch

import sim.gvars as gvars
import sim.time
from sim.aggregator import Aggregator
from sim.fl_client import FederatedLearningClient
from sim.hospital import Hospital
from sim.transport import InProcessTransport, inprocess_address_space
from training import trainer
from training.xray import xray_data, xray_training
from training.xray.xray_data import XrayDataset
from training.xray.xray_params import XrayParams, PHASES as XRAY_PHASES
from util.dp_compat import convert_dp_state_dict
from util.timer import Timer
from util.utils import dirichlet_partitions, auto_type
from util.weights import Weights

warnings.filterwarnings("ignore", category=UserWarning, message=r"Full backward hook is firing .*")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Secure RNG turned off.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Optimal order is the.*")

logger = logging.getLogger(__name__)

logging.getLogger("opacus.validators.batch_norm").setLevel("WARNING")
logging.getLogger("opacus.validators.module_validator").setLevel("WARNING")


def create_participants(num_hospitals: int, seed: int = 0, devices=None) -> List[Hospital]:
	hospitals = []
	partitions = list(dirichlet_partitions(range(xray_data.TRAIN_SIZE), num_hospitals, alpha=gvars.fl_params.dirichlet_alpha, seed=seed))
	logger.info(f"Hospital dataset partitions: {[len(p) for p in partitions]}")
	for i, p in enumerate(partitions):
		hosp = Hospital(f"Hospital {i}", device=devices[i % len(devices)] if devices is not None else "cuda")
		hosp.add_project('cxr', FederatedLearningClient(
			hosp.name,
			XrayDataset(offset=p.start, size=len(p)),
		))
		hospitals.append(hosp)
		inprocess_address_space[hosp.name] = hosp.fl_projects['cxr']
		gvars.fl_clients[hosp.name] = hosp.fl_projects['cxr']
	return hospitals


def initialize_participants(hospitals: List[Hospital], aggregator: Aggregator, phase='testing', seed=0, **kwargs):
	for hosp in hospitals:
		torch.manual_seed(seed)
		params = XrayParams(phase, device=hosp.device, **kwargs)
		model = params.get_model()
		client = hosp.fl_projects['cxr']
		client.dataset.transform = params.get_transform()  # before set_params: workers snapshot the dataset when they spawn
		client.set_params(params)
		establish_connection(client, aggregator)
		client.start()


def update_params(hospitals: List[Hospital], phase, **kwargs):
	for hosp in hospitals:
		flp = hosp.fl_projects['cxr']
		params = XrayParams(phase, device=hosp.device, **kwargs)
		# refresh params and models to new phase, only keep weights
		params._weights = convert_dp_state_dict(flp.model.state_dict())
		flp.trained_epochs = 0
		flp.dataset.transform = params.get_transform()  # before set_params: workers snapshot the dataset
		flp.set_params(params)


def setup_federation(hospitals: List[Hospital]):
	join_tasks = [asyncio.create_task(hosp.fl_projects['cxr'].join_federation()) for hosp in hospitals]
	return asyncio.gather(*join_tasks)


def establish_connection(client: FederatedLearningClient, aggregator: Aggregator):
	transport = InProcessTransport(latency=0.5)
	client.connect(transport.create_socket(), 'aggregator')
	aggregator.connect(transport.create_socket())


async def run_simulation(num_participants=3, phases=['testing'], checkpoint=None, devices=['cuda'], seed=0, **extra_params):
	event_loop = asyncio.get_event_loop()
	event_loop.slow_callback_duration = 15.0

	torch.manual_seed(seed)
	device_count = torch.cuda.device_count()
	devices = [f"cuda:{i}" for i in range(device_count)] if devices is None else devices

	xray_data.setup_shm()

	hospitals = create_participants(num_participants, devices=devices, seed=seed)
	aggregator = Aggregator()
	gvars.aggregator = aggregator
	inprocess_address_space['aggregator'] = aggregator
	total_time = 0

	for phase_i, phase in enumerate(phases):
		params = dict(checkpoint=checkpoint, save=False) | extra_params
		save_params = {}

		epochs_remaining = XRAY_PHASES[phase]['epochs']

		logger.info(f"\n\n\nStarting phase {phase} with {epochs_remaining} epochs\n\n\n")

		if phase_i == 0:
			aggregator.start()
			logger.info("Aggregator started")
			initialize_participants(hospitals, aggregator, phase=phase, seed=seed, **params)
			logger.info("Clients initialized")
			logger.info(f"Params: {hospitals[0].fl_projects['cxr'].training_params}")
			model0 = hospitals[0].fl_projects['cxr'].model
			aggregator.weight_deltas[-1] = Weights(model0) * 0.0
			aggregator.weight_deltas[0] = Weights(model0) * 1.0
			del model0
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

		training_params = hospitals[-1].fl_projects['cxr'].training_params
		test_params = XrayParams(**(params | dict(resolution=training_params.resolution, batch_size=training_params.batch_size, device=training_params.device)))
		test_dataset = XrayDataset(offset=xray_data.TEST_OFFSET, size=xray_data.TEST_SIZE, transform=test_params.get_transform())
		test_loader = xray_training.make_test_loader(test_params, test_dataset, num_workers=int(os.getenv("FL_DATA_LOADER_WORKERS", 1)))

		sim.time.init_time()

		rounds = (epochs_remaining + gvars.fl_params.epochs_per_round - 1) // gvars.fl_params.epochs_per_round
		for round in range(rounds):
			logger.info("Starting round %d", round)

			timer = Timer(print=False)
			with timer:
				await aggregator.start_new_round()
				await aggregator.round_end_event.wait()
			await asyncio.sleep(1)
			await asyncio.sleep(1)
			logger.info(f"Round done, took {timer.elapsed} seconds")
			total_time += timer.elapsed

			# models = [hosp.fl_projects['cxr'].model for hosp in hospitals]

			# assert all((Weights(models[0].cpu()) == Weights(m.cpu()) for m in models)), "Models are not synchronized after round" # side effects

			logs['time'].append([hosp.fl_projects['cxr'].training_times.copy() for hosp in hospitals])
			logs['loss'].append([hosp.fl_projects['cxr'].training_losses.copy() for hosp in hospitals])
			try:
				acc = trainer.test(hospitals[-1].fl_projects['cxr'].model, test_params, test_loader)
				logger.info(f"Model acc: {acc}")
				logs['acc'].append(acc)
			except Exception as e:
				logger.error("Error during model testing")
				logger.error(e)

			save_params = hospitals[-1].fl_projects['cxr'].training_params.__dict__.copy()

		logger.info(f"Phase {phase} done")
		# models = [hosp.fl_projects['cxr'].model for hosp in hospitals]

		try:
			logger.info("Saving model")
			save_params.update(
				phase=f"fl_{phase}",
				argv=sys.argv)
			trainer.save(hospitals[-1].fl_projects['cxr'].model, save_params, logs)
			logger.debug("Saving model done")
		except Exception:
			logger.error("Exception occured during saving")
			import traceback
			traceback.print_exc(file=sys.stdout)

	logger.info("All phases done, shutting down")
	logger.info(f"Total simulation time: {total_time}")
	aggregator.shutdown()
	for hosp in hospitals:
		hosp.fl_projects['cxr'].shutdown()

	logger.info("Simulation done")


def parse_args():
	import argparse
	parser = argparse.ArgumentParser(description="Run Federated Learning Simulation")
	parser.add_argument("--num-participants", "-n", type=int, default=3, help="Number of participants in the simulation")
	parser.add_argument("--epochs-per-round", "-r", type=int, default=gvars.fl_params.epochs_per_round, help="Number of epochs per round")
	parser.add_argument("--phase", "-p", action='append', type=str, default=[], help="Training phases for parameter presets")
	parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Model checkpoint to load")
	parser.add_argument("--name", type=str, default=trainer.save_name, help="Name to use in save path template")
	parser.add_argument("--save-path", "-o", type=str, default=trainer.save_model_path, help="Path template for checkpoint output")
	parser.add_argument("--logs-path", type=str, default=trainer.save_logs_path, help="Path template for logs output")
	parser.add_argument("--devices", "-d", action='append', type=str, default=None, help="Devices to use for participants (default: all available GPUs)")
	parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
	parser.add_argument("--use-smpc", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=gvars.fl_params.use_smpc, help="Use SMPC for secure aggregation")
	parser.add_argument("--round-timeout", type=float, default=gvars.fl_params.round_timeout, help="Time limit for each round")
	parser.add_argument("--debug", action='store_true', help="Enable debug logging")
	parser.add_argument("--dirichlet-alpha", type=float, default=gvars.fl_params.dirichlet_alpha, help="Dirichlet alpha for dataset partitioning")
	parser.add_argument("--latency", "-L", type=float, default=gvars.fl_params.latency, help="Simulated network latency")
	parser.add_argument("--reset-opt", type=lambda x: x.lower() in ['true', 'yes'], choices=[True, False], default=gvars.fl_params.reset_opt, help="Reset local optimizer states after every round")
	parser.add_argument("-P", action='append', default=[], help="Override training parameters")
	parser.add_argument("--max-norm", type=float, default=gvars.fl_params.max_norm, help="Max delta norm for clipping")
	return parser.parse_args()


def main():
	args = parse_args()
	trainer.save_name = args.name
	trainer.save_model_path = args.save_path
	trainer.save_logs_path = args.logs_path
	gvars.fl_params.use_smpc = args.use_smpc
	gvars.fl_params.epochs_per_round = args.epochs_per_round
	gvars.fl_params.round_timeout = args.round_timeout
	gvars.fl_params.dirichlet_alpha = args.dirichlet_alpha
	gvars.fl_params.latency = args.latency
	gvars.fl_params.reset_opt = args.reset_opt
	gvars.fl_params.max_norm = args.max_norm

	if len(args.phase) == 0:
		args.phase = ['testing']
	if not args.devices:
		args.devices = None  # run_simulation then spreads clients over all visible GPUs

	extra_params = {k: auto_type(v, k) for (k, v) in [p.partition('=')[::2] for p in args.P]}

	assert (('target_epsilon' in extra_params) == ('target_delta' in extra_params) == ('grad_norm' in extra_params)
	        ) or (('grad_norm' in extra_params) == ('noise_mult' in extra_params))

	logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, stream=sys.stdout, force=True)
	logging.captureWarnings(True)  # warnings.warn writes to stderr by default
	# hang forensics: `kill -USR2 <pid>` (or scancel --signal=USR2) dumps all thread stacks;
	# HANG_DUMP_INTERVAL=<secs> additionally dumps them periodically without being asked
	faulthandler.register(signal.SIGUSR2, file=sys.stdout, all_threads=True)
	if os.getenv("HANG_DUMP_INTERVAL"):
		faulthandler.dump_traceback_later(int(os.getenv("HANG_DUMP_INTERVAL")), repeat=True, file=sys.stdout)
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
