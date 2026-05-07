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
	async def send(self, msg: Message):
		pass

	@abstractmethod
	async def recv(self):
		pass


class InProcessTransportSocket(TransportSocket):
	transport: InProcessTransport
	queue: asyncio.Queue

	def __init__(self, transport):
		super().__init__()
		self.transport = transport
		self.queue = asyncio.Queue()

	async def put(self, msg: Message):
		await self.queue.put(msg)

	@override
	async def send(self, msg: Message):
		await self.transport.put(msg, self)

	@override
	async def recv(self):
		return await self.queue.get()


class InProcessTransport:
	endpoints: List[InProcessTransportSocket]
	latency: float

	def __init__(self, latency=0.):
		self.endpoints = []
		self.latency = latency

	async def put(self, msg: Message, source: InProcessTransportSocket):
		await asyncio.sleep(self.latency)
		await asyncio.gather(*[sock.put(msg) for sock in self.endpoints if sock is not source])

	def create_socket(self):
		sock = InProcessTransportSocket(self)
		self.endpoints.append(sock)
		return sock
