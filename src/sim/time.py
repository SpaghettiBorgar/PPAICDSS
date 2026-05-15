import time

sim_start_time: float = 0


def init_time():
	global sim_start_time
	sim_start_time = time.time()


def get_time() -> float:
	return time.time() - sim_start_time
