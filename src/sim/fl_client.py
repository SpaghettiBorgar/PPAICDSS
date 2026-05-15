import asyncio
import logging
import sys
import time
import traceback
from typing import List

import torch.nn

from sim.messages import *
from sim.transport import TransportSocket
from training import training
from training.params import Params
from training.xray import xray_training

logger = logging.getLogger(__name__)


class FederatedLearningClient:
	client_id: ClientID
	dataset: torch.utils.data.Dataset
	model: torch.nn.Module
	local_rev: ModelRev
	global_rev: ModelRev
	global_weights: Weights
	aggregator: TransportSocket
	current_round: Round
	waiting_for_update: bool
	training: bool
	training_task: asyncio.Task
	training_params: Params
	training_losses: List[List[float]]
	training_times: List[float]
	loop_task: asyncio.Task

	def __init__(self, client_id, dataset):
		self.client_id = client_id
		self.dataset = dataset
		self.model = None
		self.local_rev = 0
		self.global_weights = None
		self.waiting_for_update = False
		self.training = False
		self.training_losses = []
		self.training_times = []

	def start(self):
		self.loop_task = asyncio.create_task(self.loop())

	def shutdown(self):
		logger.info("[%s] shutting down", self.client_id)
		self.loop_task.cancel()

	async def loop(self):
		logger.info("Client %s running", self.client_id)
		while True:
			msg = await self.aggregator.recv()
			logger.debug("[%s] %s", self.client_id, msg)
			await self.handle_message(msg)

	def set_params(self, params):
		self.training_params = params

	def start_training(self):
		self.training = True
		self.training_task = asyncio.create_task(asyncio.to_thread(self.do_training_step))
		self.training_task.add_done_callback(lambda t: asyncio.create_task(self.on_training_complete(t)))

	async def on_training_complete(self, task: asyncio.Task):
		self.training = False
		exc = task.exception()
		if exc is not None:
			traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
		else:
			await self.aggregator.send(DeltaPush(WeightsDelta(
				rev_a=self.current_round.rev_a,
				rev_b=self.current_round.rev_b,
				diff=self.get_delta()
			)))

	def do_training_step(self):
		params = self.training_params
		logger.info("%s starting training", self.client_id)
		data_loader = xray_training.make_train_loader(params, self.dataset, batch_size=params.batch_size, drop_last=True)
		for epoch in range(params.epochs):
			logger.debug("%s epoch %d", self.client_id, epoch)
			start_time = time.time()
			self.training_losses.append(training.train(self.model, params, data_loader, epoch=epoch))
			self.training_times.append(time.time() - start_time)

	def apply_delta(self, delta: WeightDiff):
		Weights(self.model).add(delta)

	def get_delta(self) -> Weights:
		if self.global_weights is None:
			raise InvalidStateException()
		return Weights(self.model) - self.global_weights

	def set_weights(self, weights: Weights):
		Weights(self.model).assign(weights)

	def set_model(self, model: torch.nn.Module):
		self.model = model
		self.global_weights = Weights(self.model) * 0

	async def update_if_necessary(self):
		if self.local_rev < self.global_rev:
			await self.aggregator.send(UpdateRequest(rev_a=self.local_rev, rev_b=0))
			return True
		else:
			return False

	async def handle_message(self, msg: Message):
		match msg:
			case FederationResponse():
				await self.handle_federation_response(msg)
			case DeltaPush():
				await self.handle_delta_push(msg)
			case RoundAnnounce():
				await self.handle_round_announce(msg)
			case RoundEnd():
				self.handle_round_end(msg)

	async def handle_round_announce(self, msg: RoundAnnounce):
		self.current_round = msg.round
		self.global_rev = msg.round.rev_a
		if await self.update_if_necessary():
			self.waiting_for_update = True
		else:
			self.start_training()

	def handle_round_end(self, msg: RoundEnd):
		pass

	async def handle_federation_response(self, msg: FederationResponse):
		self.global_rev = msg.global_model_rev
		await self.update_if_necessary()

	async def handle_delta_push(self, msg: DeltaPush):
		if self.local_rev != msg.delta.rev_a or msg.delta.rev_b < self.global_rev:
			raise InvalidStateException()
		self.apply_delta(msg.delta.diff)
		self.global_rev = msg.delta.rev_b
		self.local_rev = msg.delta.rev_b

		logger.info("[%s] local_rev now %d", self.client_id, self.local_rev)
		if self.waiting_for_update:
			self.waiting_for_update = False
			self.start_training()

	def connect(self, sock: TransportSocket):
		self.aggregator = sock

	async def join_federation(self):
		logger.info("[%s] joining federation", self.client_id)
		await self.aggregator.send(FederationRequest(client_id=self.client_id, local_model_rev=self.local_rev))
