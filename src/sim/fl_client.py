import asyncio
import logging
import sys
import time
import traceback
from typing import List

import torch.nn

import sim.gvars as gvars
from sim.messages import *
from sim.transport import InProcessTransportSocket, TransportSocket
from training import training
from training.params import Params
from training.xray import xray_training
from util import smpc
from util.utils import make_seed
from util.dp_compat import make_private_auto

MODEL_SHAPES = {}


class FederatedLearningClient:
	client_id: ClientID
	logger: logging.Logger
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
	join_event: asyncio.Event
	rng: np.random.Generator
	key: KeyShare
	key_shares: dict[ClientID, KeyShare]
	received_shares: dict[ClientID, KeyShare]
	key_phase_group: List[ClientID]
	peers: dict[ClientID, TransportSocket]
	listeners: set[asyncio.Task]
	trained_epochs: int

	def __init__(self, client_id, dataset):
		self.client_id = client_id
		self.logger = logging.getLogger(client_id)
		self.dataset = dataset
		self.model = None
		self.local_rev = 0
		self.global_weights = None
		self.waiting_for_update = False
		self.training = False
		self.training_losses = []
		self.training_times = []
		self.rng = np.random.default_rng(seed=make_seed(client_id))
		self.key_shares = {}
		self.received_shares = {}
		self.peers = {}
		self.listeners = set()
		self.trained_epochs = 0
		self.key_phase_group = []
		self.join_event = asyncio.Event()

	def start(self):
		self.loop_task = asyncio.create_task(self.loop())

	def shutdown(self):
		self.logger.info("Shutting down")
		self.loop_task.cancel()

	async def loop(self):
		self.logger.info("Client %s running", self.client_id)
		try:
			await asyncio.Future()
		except asyncio.CancelledError:
			for task in self.listeners:
				task.cancel()
			await asyncio.gather(*self.listeners, return_exceptions=True)
			raise

	async def _listen(self, sock: TransportSocket):
		while True:
			msg = await sock.recv()
			await self.handle_message(msg, sock)

	def set_params(self, params, init=True):
		if init:
			model = params.get_model()
			data_loader = xray_training.make_train_loader(params, self.dataset, batch_size=params.batch_size, drop_last=False)
			optimizer = params.get_optimizer()
			params._model, params._optimizer, self.data_loader = make_private_auto(model, optimizer, data_loader, params)
			self.set_model(params._model)
		else:
			self.model = params._model
		params.log_prefix = f"[{self.client_id}] "
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
			if gvars.fl_params.use_smpc:
				await self.push_encrypted_update()
			else:
				self.aggregator.send(DeltaPush(WeightsDelta(
					rev_a=self.current_round.rev_a,
					rev_b=self.current_round.rev_b,
					diff=self.get_clipped_delta()
				)))

	async def push_encrypted_update(self):
		self.key = smpc.generate_key_mask(self.rng)
		delta = self.get_delta()
		vec, shapes = delta.flatten()
		norm = np.linalg.norm(vec)
		self.logger.debug(f"[clipping] delta is norm={norm}, range={vec.min()}_{vec.max()}, mean={vec.mean()}, std={vec.std()}")
		vec = smpc.quantize(smpc.clip(vec))
		crypt_vec = smpc.encrypt(vec, self.key)
		self.aggregator.send(EncryptedDeltaPush(EncryptedWeightsDelta(
			rev_a=self.current_round.rev_a,
			rev_b=self.current_round.rev_b,
			diff=crypt_vec
		)))

	def do_training_step(self):
		params = self.training_params
		epochs = min(gvars.fl_params.epochs_per_round, params.epochs - self.trained_epochs)
		self.logger.info(f"starting training for {epochs} epochs")
		for epoch in range(epochs):
			epoch += self.trained_epochs
			self.logger.debug(" Training epoch %d", epoch)
			start_time = time.time()
			self.training_losses.append(training.train(self.model, params | dict(epochs=gvars.fl_params.epochs_per_round), self.data_loader, epoch=epoch))
			self.training_times.append(time.time() - start_time)
			self.trained_epochs += 1

	def apply_delta(self, delta: WeightDiff):
		Weights(self.model).add(delta)

	def get_delta(self) -> Weights:
		if self.global_weights is None:
			raise InvalidStateException()
		return Weights(self.model) - self.global_weights

	def get_clipped_delta(self) -> Weights:
		delta = self.get_delta()
		vec, shapes = delta.flatten()
		vec = smpc.clip(vec)
		return Weights.unflatten(vec, shapes)

	def set_weights(self, weights: Weights):
		Weights(self.model).assign(weights)

	def set_model(self, model: torch.nn.Module):
		self.model = model
		weights = Weights(self.model)
		MODEL_SHAPES[self.client_id] = weights.shapes()
		self.global_weights = weights * 0

	async def update_if_necessary(self):
		if self.local_rev < self.global_rev:
			self.aggregator.send(UpdateRequest(rev_a=self.local_rev, rev_b=0))
			return True
		else:
			return False

	async def handle_message(self, msg: Message, src: TransportSocket = None):
		src_id = next((peer_id for peer_id, s in self.peers.items() if s is src), 'aggregator' if src is self.aggregator else 'unknown')
		self.logger.debug(f"Received from %s: %s", src_id, msg)
		match msg:
			case FederationResponse():
				await self.handle_federation_response(msg)
			case DeltaPush():
				await self.handle_delta_push(msg)
			case RoundAnnounce():
				await self.handle_round_announce(msg)
			case KeyPhaseAnnounce():
				await self.handle_key_phase_announce(msg)
			case RoundEnd():
				self.handle_round_end(msg)
			case SMPCKeyShare():
				await self.handle_smpc_key_share(msg, next((peer_id for peer_id, sock in self.peers.items() if sock is src)))
			case Ping():
				if not msg.is_reply:
					src.send(Ping(is_reply=True))
			case _:
				raise InvalidStateException(f"Unexpected message type {type(msg)}")

	async def handle_round_announce(self, msg: RoundAnnounce):
		self.current_round = msg.round
		self.global_rev = msg.round.rev_a
		self.key_shares = {}
		self.received_shares = {}
		self.key_phase_group = []
		if await self.update_if_necessary():
			self.waiting_for_update = True
		else:
			self.start_training()

	async def do_peer_exchange(self):
		for peer_id, sock in self.peers.items():
			if not peer_id in self.key_shares:
				self.key_shares[peer_id] = smpc.generate_key_mask(self.rng)
			self.logger.info(f"Sending key share to {peer_id}")
			sock.send(SMPCKeyShare(key_share=self.key_shares[peer_id]))
		if len(self.received_shares) == len(self.key_phase_group) != 0:
			await self.submit_share_sum()

	async def handle_smpc_key_share(self, msg: SMPCKeyShare, peer_id):
		self.received_shares[peer_id] = msg.key_share
		self.logger.debug(f"{len(self.received_shares)}/{len(self.key_phase_group)} key shares received")
		if len(self.received_shares) == len(self.key_phase_group) != 0:
			await self.submit_share_sum()

	async def submit_share_sum(self):
		self.logger.info("Submitting key share sum to aggregator")
		server_share = smpc.make_last_share(self.key, list(self.key_shares.values()))
		server_share = (server_share + sum(self.received_shares.values())) % smpc.MOD
		self.aggregator.send(SMPCKeyShare(key_share=server_share))

	async def handle_key_phase_announce(self, msg: KeyPhaseAnnounce):
		self.key_phase_group = msg.group
		self.received_shares[self.client_id] = self.key_shares[self.client_id] = smpc.generate_key_mask(self.rng)
		for client_id in msg.group:
			if client_id != self.client_id and client_id not in self.peers:
				self.connect(InProcessTransportSocket.connect_to(dest=client_id, source=self.client_id), client_id)
		await self.do_peer_exchange()

	def handle_round_end(self, msg: RoundEnd):
		if msg.success:
			if msg.round != self.current_round or self.local_rev != msg.delta.rev_a:
				raise InvalidStateException()
		self.apply_delta(msg.delta.diff)
		self.local_rev = msg.delta.rev_b

	async def handle_federation_response(self, msg: FederationResponse):
		self.global_rev = msg.global_model_rev
		await self.update_if_necessary()
		self.join_event.set()

	async def handle_delta_push(self, msg: DeltaPush):
		if self.local_rev != msg.delta.rev_a or msg.delta.rev_b < self.global_rev:
			raise InvalidStateException()
		self.apply_delta(msg.delta.diff)
		self.global_rev = msg.delta.rev_b
		self.local_rev = msg.delta.rev_b

		self.logger.debug("local_rev now %d", self.local_rev)
		if self.waiting_for_update:
			self.waiting_for_update = False
			self.start_training()

	def connect(self, sock: TransportSocket, id):
		if id == 'aggregator':
			self.aggregator = sock
		else:
			self.peers[id] = sock
		task = asyncio.create_task(self._listen(sock))
		self.listeners.add(task)
		task.add_done_callback(self.listeners.discard)

	async def join_federation(self):
		self.logger.info("joining federation")
		self.aggregator.send(FederationRequest(client_id=self.client_id, local_model_rev=self.local_rev))
		await self.join_event.wait()
