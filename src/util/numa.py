import os

import torch


def _parse_cpulist(text: str) -> set:
	cpus = set()
	for part in text.strip().split(','):
		if part:
			lo, _, hi = part.partition('-')
			cpus.update(range(int(lo), int(hi or lo) + 1))
	return cpus


def device_local_cpus(device) -> set | None:
	"""CPUs on the NUMA node(s) attached to a CUDA device's PCIe root complex,
	restricted to the CPUs this process is actually allowed to run on (cgroup/SLURM cpuset).
	Returns None if the device is not a GPU or locality can't be determined."""
	try:
		device = torch.device(device)
		if device.type != 'cuda' or not torch.cuda.is_available():
			return None
		props = torch.cuda.get_device_properties(device)
		pci_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
		with open(f"/sys/bus/pci/devices/{pci_id}/local_cpulist") as f:
			local = _parse_cpulist(f.read())
	except (OSError, AttributeError, RuntimeError):
		return None
	local &= os.sched_getaffinity(0)
	return local or None


def bind_current_thread(device) -> set | None:
	"""Pin the calling thread (sched_setaffinity(0) is per-thread on Linux) to the
	device-local CPUs. No-op when locality is unknown or outside our cpuset."""
	cpus = device_local_cpus(device)
	if cpus:
		os.sched_setaffinity(0, cpus)
	return cpus


class WorkerInit:
	"""DataLoader worker_init_fn: binds the worker to its device's NUMA domain so
	decompression runs — and shm cache pages get first-touched — GPU-locally.
	A class instead of a closure so it stays picklable under any mp start method."""

	def __init__(self, device=None):
		self.cpus = device_local_cpus(device) if device is not None else None

	def __call__(self, worker_id):
		import faulthandler
		faulthandler.enable()
		if self.cpus:
			os.sched_setaffinity(0, self.cpus)
		torch.set_num_threads(1)
		print(f"Worker {worker_id} initialized with affinity {os.sched_getaffinity(0)}")
