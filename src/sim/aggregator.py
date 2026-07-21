import asyncio
import logging
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, OrderedDict, override

import sim.gvars as gvars
from sim.messages import *
from sim.time import get_time
from sim.transport import TransportSocket
from util import smpc, aio
from util.utils import chunk_with_min_remainder

logger = logging.getLogger("Aggregator")


class AggregationException(Exception):
	pass


class AggregationStrategy(ABC):
	@abstractmethod
	def aggregate_deltas(self, deltas: List[WeightDiff]) -> WeightDiff:
		pass


class FedAvg(AggregationStrategy):
	@override
	def aggregate_deltas(self, deltas: List[WeightDiff]) -> WeightDiff:
		return sum_(deltas) / len(deltas)


@dataclass
class ClientState:
	socket: TransportSocket
	model_rev: ModelRev
	last_ping: float = 0


def sum_(*args):
	return reduce(operator.add, *args)


class Aggregator:
	clients: dict[ClientID, ClientState]
	connections: dict[TransportSocket, ClientID | None]
	listeners: set[asyncio.Task]
	global_model_rev: ModelRev
	current_round: Round | None
	weight_deltas: OrderedDict[ModelRev, WeightDiff]
	current_round_deltas: dict[ClientID, WeightDiff] | dict[ClientID, EncryptedWeights]
	aggregation_strategy: AggregationStrategy
	loop_task: asyncio.Task
	round_end_event: asyncio.Event
	round_deadline_task: asyncio.Task | None
	smpc_groups: List[List[ClientID]]
	smpc_key_shares: dict[ClientID, KeyShare]
	ending_round: bool
	aggregating: bool

	def __init__(self):
		self.clients = {}
		self.connections = {}
		self.global_model_rev = 0
		self.weight_deltas = OrderedDict()
		self.aggregation_strategy = FedAvg()
		self.current_round = None
		self.listeners = set()
		self.round_end_event = asyncio.Event()
		self.round_deadline_task = None
		self.ending_round = False
		self.aggregating = False

	def start(self):
		self.loop_task = asyncio.create_task(self.loop())

	def shutdown(self):
		logger.info("Aggregator shutting down")
		if self.round_deadline_task is not None:
			self.round_deadline_task.cancel()
		try:
			self.loop_task.cancel()
		except AttributeError:
			logger.warning("Aggregator wasn't running")

	async def loop(self):
		logger.info("Aggregator running")
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
			logger.debug("Received from %s: %s" % (self.connections.get(sock), msg))
			try:
				await self.handle_message(msg, sock)
			except Exception:
				logger.exception(f"Error handling {type(msg).__name__} from {self.connections.get(sock)}; message dropped, still listening")

	async def handle_message(self, msg: Message, src: TransportSocket):
		if src in self.connections:
			client_id = self.connections[src]
		elif not isinstance(msg, FederationRequest):
			raise InvalidStateException()

		match msg:
			case FederationRequest():
				await self.handle_federation_request(msg, src=src)
			case UpdateRequest():
				await self.handle_update_request(msg, client_id)
			case DeltaPush():
				await self.handle_delta_push(msg, client_id)
			case SMPCKeyShare():
				await self.handle_smpc_key_share(msg, client_id)
			case Ping():
				self.clients[client_id].last_ping = get_time()
				if not msg.is_reply:
					src.send(Ping(is_reply=True))
			case _:
				raise InvalidStateException(f"Unexpected message type {type(msg)}")

	async def handle_federation_request(self, req: FederationRequest, src: TransportSocket):
		if req.client_id in self.clients:
			raise InvalidStateException()
		self.clients[req.client_id] = ClientState(socket=src, model_rev=req.local_model_rev)
		self.connections[src] = req.client_id
		self.clients[req.client_id].socket.send(FederationResponse(global_model_rev=self.global_model_rev))

	async def handle_update_request(self, req: UpdateRequest, client: ClientID):
		state = self.clients[client]
		if req.rev_b == 0:
			req.rev_b = self.global_model_rev
		state.socket.send(DeltaPush(WeightsDelta(req.rev_a, req.rev_b, self.build_delta(req.rev_a, req.rev_b))))
		state.model_rev = req.rev_b

	async def handle_delta_push(self, req: DeltaPush, client: ClientID):
		self.clients[client].model_rev = req.delta.rev_b
		assert self.current_round is not None
		if (req.delta.rev_a, req.delta.rev_b) != (self.current_round.rev_a, self.current_round.rev_b):
			raise InvalidStateException()
		self.current_round_deltas[client] = req.delta.diff
		self.check_round_end_condition()

	def check_round_end_condition(self):
		if self.current_round is None or self.ending_round:
			return
		num_deltas = len(self.current_round_deltas)
		if num_deltas == len(self.clients):
			self.begin_round_end()
		elif get_time() >= self.current_round.deadline:
			self.begin_round_end(timeout=True)

	def begin_round_end(self, timeout=False):
		if timeout:
			logger.info("Timeout reached.")
		if self.current_round is None or self.ending_round:
			return
		self.ending_round = True
		missing_clients = [c_id for c_id in self.clients.keys() if c_id not in self.current_round_deltas]
		logger.info(f"Ending round. Missing clients: {missing_clients}")
		if self.round_deadline_task is not None:
			self.round_deadline_task.cancel()
			self.round_deadline_task = None
		aio.spawn(self.end_round(), logger=logger, name="end_round")

	async def watch_round_deadline(self, round_id: int, deadline: Timestamp):
		try:
			await asyncio.sleep(max(0, deadline - get_time()))
			if self.current_round is not None and self.current_round.round_id == round_id:
				self.begin_round_end(timeout=True)
		except asyncio.CancelledError:
			raise

	async def start_key_phase(self):
		self.smpc_groups = []
		self.smpc_key_shares = {}
		for client_id, client_state in self.clients.items():
			if client_id in self.current_round_deltas:
				client_state.socket.send(Ping())
		await asyncio.sleep(0)
		await asyncio.sleep(4)
		threshold = get_time() - 5
		surviving_clients = [c_id for c_id, c_state in self.clients.items() if c_id in self.current_round_deltas and c_state.last_ping > threshold]
		logger.info("Surviving clients: %s", surviving_clients)
		self.smpc_groups = chunk_with_min_remainder(surviving_clients, n=3, n_min=2)
		logger.info("Starting key phase with groups %s", self.smpc_groups)
		if not self.smpc_groups:
			# no shares will ever arrive, so nothing would trigger aggregate()
			await self.aggregate()
			return
		for group in self.smpc_groups:
			for client_id in group:
				self.clients[client_id].socket.send(KeyPhaseAnnounce(group=group))
		aio.spawn(self.watch_key_phase(self.current_round.round_id), logger=logger, name="key phase watchdog")

	async def watch_key_phase(self, round_id: int):
		try:
			await asyncio.wait_for(self.round_end_event.wait(), timeout=gvars.fl_params.key_phase_timeout)
		except TimeoutError:
			if self.current_round is not None and self.current_round.round_id == round_id:
				logger.warning("Key phase timed out, aggregating with available shares")
				await self.aggregate()

	async def handle_smpc_key_share(self, msg: SMPCKeyShare, client_id):
		if not any(client_id in g for g in self.smpc_groups):
			logger.warning("Ignoring key share from %s: not in any group", client_id)
			return
		self.smpc_key_shares[client_id] = msg.key_share
		if len(self.smpc_key_shares) == sum(len(g) for g in self.smpc_groups):
			aio.spawn(self.aggregate(), logger=logger, name="aggregate")

	async def aggregate(self):
		if self.aggregating:
			return
		self.aggregating = True
		assert self.current_round is not None
		logger.info(f"Aggregating {len(self.current_round_deltas)} deltas")
		msg = None
		try:
			if gvars.fl_params.use_smpc:
				deltas = []
				recovered = 0
				for group in self.smpc_groups:
					try:
						group_key = sum(self.smpc_key_shares[c_id] for c_id in group) % smpc.MOD
						group_sum = smpc.decrypt(sum([self.current_round_deltas[c_id] for c_id in group]) % smpc.MOD, group_key)
						# decoding yields the SUM over group members; spread the group mean over
						# len(group) entries so FedAvg over entries equals per-client averaging
						group_mean = smpc.unquantize(smpc.recover_sign(group_sum)) / len(group)
						deltas.extend([group_mean] * len(group))
						recovered += 1
					except KeyError:
						logger.warning("Not all key shares received for group %s, skipping", group)
				logger.info(f"Successfully recovered {recovered} out of {len(self.smpc_groups)} group deltas")
			else:
				deltas = list(self.current_round_deltas.values())

			if len(deltas) == 0:
				raise AggregationException("No deltas available for aggregation")
			aggregate = self.aggregation_strategy.aggregate_deltas(deltas)
			agg_vec = aggregate.flatten()[0] if isinstance(aggregate, Weights) else aggregate
			logger.debug(f"aggregate norm={np.linalg.norm(agg_vec)} range={agg_vec.min()}_{agg_vec.max()} mean={agg_vec.mean()} std={agg_vec.std()}")
			msg = RoundEnd(round=self.current_round, success=True,
			               delta=WeightsDelta(rev_a=self.current_round.rev_a, rev_b=self.current_round.rev_b, diff=aggregate))
			self.weight_deltas[self.current_round.rev_b] = aggregate
			# memory optimization: drop per-round deltas that every client has advanced past
			# (but always keep the last two; build_delta raises if a catch-up needs a pruned rev)
			min_rev = min((state.model_rev for state in self.clients.values()), default=self.current_round.rev_b)
			for old_rev in [rev for rev in self.weight_deltas if rev < min(min_rev, self.current_round.rev_b - 1)]:
				del self.weight_deltas[old_rev]
			self.global_model_rev = self.current_round.rev_b
		except Exception:
			logger.exception("Aggregation failed, declaring round failed")
		finally:
			if msg is None:
				msg = RoundEnd(round=self.current_round, success=False, delta=None)
			for client_state in self.clients.values():
				client_state.socket.send(msg)
			if self.round_deadline_task is not None:
				self.round_deadline_task.cancel()
				self.round_deadline_task = None
			self.round_end_event.set()

	async def end_round(self):
		try:
			if gvars.fl_params.use_smpc:
				await self.start_key_phase()
			else:
				await self.aggregate()
		except Exception:
			logger.exception("end_round failed, declaring round failed")
			await self.aggregate()  # no-op if aggregation already ran; otherwise fails the round and releases waiters

	def build_delta(self, rev_a: ModelRev, rev_b: ModelRev):
		revs = [d for d in self.weight_deltas if rev_a < d <= rev_b]
		if len(revs) != rev_b - rev_a:
			raise InvalidStateException(
				f"Cannot build delta {rev_a}->{rev_b}: missing revisions, have {sorted(self.weight_deltas)}")
		return sum_((self.weight_deltas[d] for d in revs))

	async def start_new_round(self):
		if len(self.clients) == 0:
			logger.error("No clients connected, cannot start new round")
			raise InvalidStateException()

		if self.round_deadline_task is not None:
			self.round_deadline_task.cancel()
			self.round_deadline_task = None
		self.round_end_event.clear()
		self.ending_round = False
		self.aggregating = False
		self.current_round = Round(
			round_id=(self.current_round.round_id if self.current_round is not None else 0) + 1,
			rev_a=self.global_model_rev,
			rev_b=self.global_model_rev + 1,
			deadline=get_time() + gvars.fl_params.round_timeout)
		logger.info("Starting round %s", self.current_round)
		self.current_round_deltas = {}
		self.round_deadline_task = asyncio.create_task(
			self.watch_round_deadline(self.current_round.round_id, self.current_round.deadline))
		for client_state in self.clients.values():
			client_state.socket.send(RoundAnnounce(round=self.current_round))

	def connect(self, sock: TransportSocket):
		logger.debug("Aggregator connected to %s", sock)
		self.connections[sock] = None
		task = aio.spawn(self._listen(sock), logger=logger, name="aggregator listen")
		self.listeners.add(task)
		task.add_done_callback(self.listeners.discard)
