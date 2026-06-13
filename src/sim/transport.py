from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias, List, override

from sim.messages import Message

ReceiveCallback: TypeAlias = Callable[[Message], None]


class TransportSocket(ABC):
	def __init__(self):
		self.callbacks = []

	@abstractmethod
	def send(self, msg: Message):
		pass

	@abstractmethod
	async def recv(self) -> Message:
		pass


inprocess_address_space = {}


class InProcessTransportSocket(TransportSocket):
	transport: InProcessTransport
	queue: asyncio.Queue

	def __init__(self, transport):
		super().__init__()
		self.transport = transport
		self.queue = asyncio.Queue()

	@classmethod
	def connect_to(cls, dest, source):
		if dest not in inprocess_address_space:
			raise ConnectionError(f"Address {dest} not found in in-process address space")
		peer = inprocess_address_space[dest]
		transport = InProcessTransport()
		sock1 = transport.create_socket()
		sock2 = transport.create_socket()
		peer.connect(sock2, source)
		return sock1

	def put(self, msg: Message):
		self.queue.put_nowait(msg)

	@override
	def send(self, msg: Message):
		self.transport.put(msg, self)

	@override
	async def recv(self) -> Message:
		return await self.queue.get()


class InProcessTransport:
	endpoints: List[InProcessTransportSocket]
	latency: float
	_dispatch_tasks: set[asyncio.Task]

	def __init__(self, latency=0.):
		self.endpoints = []
		self.latency = latency
		self._dispatch_tasks = set()

	def put(self, msg: Message, source: InProcessTransportSocket):
		task = asyncio.create_task(self._deliver(msg, source))
		self._dispatch_tasks.add(task)
		task.add_done_callback(self._dispatch_tasks.discard)

	async def _deliver(self, msg: Message, source: InProcessTransportSocket):
		if self.latency:
			await asyncio.sleep(self.latency)
		for sock in self.endpoints:
			if sock is not source:
				sock.put(msg)

	def create_socket(self):
		sock = InProcessTransportSocket(self)
		self.endpoints.append(sock)
		return sock
