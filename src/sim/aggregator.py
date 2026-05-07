import asyncio
import logging
import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, OrderedDict, override

from sim import get_time
from sim.messages import *
from sim.transport import TransportSocket

logger = logging.getLogger(__name__)

ROUND_PERIOD = 120


class AggregationException(Exception):
	pass


class AggregationStrategy(ABC):
	@abstractmethod
	def aggregate_deltas(self, deltas: List[WeightDiff]):
		pass


class FedAvg(AggregationStrategy):
	@override
	def aggregate_deltas(self, deltas: List[WeightDiff]):
		return sum_(deltas) / len(deltas)


@dataclass
class ClientState:
	socket: TransportSocket
	model_rev: ModelRev


def sum_(*args):
	return reduce(operator.add, *args)


class Aggregator:
	clients: dict[ClientID, ClientState]
	connections: dict[TransportSocket, ClientID | None]
	listeners: set[asyncio.Task]
	global_model_rev: ModelRev
	current_round: Round
	weight_deltas: OrderedDict[ModelRev, WeightDiff]
	current_round_deltas: dict[ClientID, WeightDiff]
	aggregation_strategy: AggregationStrategy
	loop_task: asyncio.Task

	def __init__(self):
		self.clients = {}
		self.connections = {}
		self.global_model_rev = 0
		self.weight_deltas = OrderedDict()
		self.aggregation_strategy = FedAvg()
		self.current_round = None
		self.listeners = set()

	def start(self):
		self.loop_task = asyncio.create_task(self.loop())

	async def loop(self):
		logger.info("Aggregator running")
		try:
			await asyncio.Future()
		except asyncio.CancelledError:
			for task in self.listeners:
				task.cancel()
			await asyncio.gather(*self.listeners, return_exceptions=True)
			raise

	async def _listen(self, sock):
		while True:
			msg = await sock.recv()
			logger.debug("Received from %s: %s", self.connections.get(sock), msg)
			await self.handle_message(msg, sock)

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

	async def handle_federation_request(self, req: FederationRequest, src: TransportSocket):
		if req.client_id in self.clients:
			raise InvalidStateException()
		self.clients[req.client_id] = ClientState(socket=src, model_rev=req.local_model_rev)
		self.connections[src] = req.client_id
		await self.clients[req.client_id].socket.send(FederationResponse(global_model_rev=self.global_model_rev))

	async def handle_update_request(self, req: UpdateRequest, client: ClientID):
		state = self.clients[client]
		if req.rev_b == 0:
			req.rev_b = self.global_model_rev
		await state.socket.send(DeltaPush(WeightsDelta(req.rev_a, req.rev_b, self.build_delta(req.rev_a, req.rev_b))))

	async def handle_delta_push(self, req: DeltaPush, client: ClientID):
		self.clients[client].model_rev = req.delta.rev_b
		if (req.delta.rev_a, req.delta.rev_b) != (self.current_round.rev_a, self.current_round.rev_b):
			raise InvalidStateException()
		self.current_round_deltas[client] = req.delta.diff
		await self.check_round_end_condition()

	async def check_round_end_condition(self):
		end = False
		num_deltas = len(self.current_round_deltas)
		if num_deltas == len(self.clients):
			end = True
		if get_time() > self.current_round.deadline:
			end = True
		if end:
			await self.end_round()

	async def end_round(self):
		try:
			aggregate = self.aggregation_strategy.aggregate_deltas(list(self.current_round_deltas.values()))
			msg = RoundEnd(round=self.current_round, success=True,
			               delta=WeightsDelta(rev_a=self.current_round.rev_a, rev_b=self.current_round.rev_b, diff=WeightDiff(aggregate)))
			self.weight_deltas[self.current_round.rev_b] = aggregate
			self.global_model_rev = self.current_round.rev_b
		except AggregationException as e:
			msg = RoundEnd(round=self.current_round, success=False, delta=None)

		for client_state in self.clients.values():
			await client_state.socket.send(msg)

	def build_delta(self, rev_a: ModelRev, rev_b: ModelRev):
		return sum_((self.weight_deltas[d] for d in self.weight_deltas if rev_a < d <= rev_b))

	async def start_new_round(self):
		self.current_round = Round(
			round_id=(self.current_round.round_id if self.current_round is not None else 0) + 1,
			rev_a=self.global_model_rev,
			rev_b=self.global_model_rev + 1,
			deadline=get_time() + ROUND_PERIOD)
		logger.info("Starting %s", self.current_round)
		self.current_round_deltas = {}
		for client_state in self.clients.values():
			await client_state.socket.send(RoundAnnounce(round=self.current_round))

	def connect(self, socket: TransportSocket):
		self.connections[socket] = None
		task = asyncio.create_task(self._listen(socket))
		self.listeners.add(task)
		task.add_done_callback(self.listeners.discard)
