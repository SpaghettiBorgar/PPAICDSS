from types import SimpleNamespace

fl_params = SimpleNamespace(
	use_smpc=True,
    epochs_per_round=2,
    round_timeout=float('inf'),
    dirichlet_alpha=10.0,
    latency=0.0
)

fl_clients = {}
aggregator = None