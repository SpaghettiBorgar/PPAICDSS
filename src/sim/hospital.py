from typing import TypeAlias, Union

import torch

from sim.fl_client import FederatedLearningClient

DeviceLikeType: TypeAlias = Union[str, torch.device, int]


class Hospital:
	fl_projects: dict[str, FederatedLearningClient]
	name: str
	device: DeviceLikeType

	def __init__(self, name, device="cuda"):
		self.name = name
		self.fl_projects = {}
		self.device = device

	def add_project(self, name: str, fl_project: FederatedLearningClient):
		self.fl_projects[name] = fl_project
