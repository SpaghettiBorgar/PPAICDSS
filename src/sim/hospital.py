from sim.fl_client import FederatedLearningClient


class Hospital:
	fl_projects: dict[str, FederatedLearningClient]
	name: str

	def __init__(self, name):
		self.name = name
		self.fl_projects = {}

	def add_project(self, name: str, fl_project: FederatedLearningClient):
		self.fl_projects[name] = fl_project
