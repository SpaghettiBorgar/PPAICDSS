from types import SimpleNamespace

fl_params = SimpleNamespace(
	use_smpc=True,
    epochs_per_round=2,
    round_timeout=float('inf')
)

fl_clients = {}