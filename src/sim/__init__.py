from __future__ import annotations

import time

sim_start_time: float = 0


def get_time() -> float:
	return time.time() - sim_start_time
