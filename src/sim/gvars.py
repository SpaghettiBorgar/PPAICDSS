from types import SimpleNamespace

fl_params = SimpleNamespace(
	epochs_per_round=1,
	use_smpc=True,
	round_timeout=20 * 60.0,  # float('inf')
	key_phase_timeout=120.0,
	dirichlet_alpha=30.0,
	latency=0.0,
	reset_opt=True,
	max_norm=40.
)

fl_clients = {}
aggregator = None
