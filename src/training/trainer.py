import json
import os
import time
from datetime import datetime
from typing import TypeAlias, Callable, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.params import Params
from util.mapping import tree_map
from util.utils import resilient_iter, fix_collate

profiler_dir = "./profiler_logs"


def make_profiler(params: Params):
	from torch.profiler import ProfilerActivity, profile, schedule

	def trace_handler(prof):
		os.makedirs(profiler_dir, exist_ok=True)
		path = os.path.join(profiler_dir, f"trace_{datetime.today().strftime('%m_%d_%H%M%S')}.json")
		prof.export_chrome_trace(path)
		sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
		print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
		print(f"{params.log_prefix}[profiler] trace saved to {path} — open it at https://ui.perfetto.dev")

	return profile(
		activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
		schedule=schedule(wait=1, warmup=2, active=8, repeat=1),
		on_trace_ready=trace_handler,
		profile_memory=True,
	)


def train(model: torch.nn.Module, params: Params, train_loader: DataLoader, epoch):
	fix_collate(train_loader)
	model.train()
	losses = []

	prof = make_profiler(params) if params.profile and epoch == 0 else None
	if prof:
		prof.start()

	# if params.grad_norm is not None and params.noise_mult is not None:
	# print("using make_model_dp_compatible with swap_blocks=True")
	# make_model_dp_compatible(model, swap_blocks=False)
	# print("using fix_inplace")

	optimizer = params.get_optimizer()
	criterion = params.get_criterion()

	for batch_idx, (inp, target) in resilient_iter(enumerate(train_loader)):
		inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
		optimizer.zero_grad()
		output = model(inp)
		loss = criterion(output, target)

		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		if prof:
			prof.step()
		if batch_idx % (1 if params.phase == 'testing' else 10) == 0:
			n_total = len(train_loader.dataset) if params.batches == 0 else min(len(train_loader.dataset), params.batches * params.batch_size)
			n_processed = min((batch_idx + 1) * params.batch_size, n_total)
			epsilon_str = ""
			if params.grad_norm is not None and params.noise_mult != 0.0:
				try:
					for delta in [params.target_delta]:
						epsilon_str = f"\tEpsilon (delta={delta}): {params.privacy_engine.get_epsilon(delta=delta)}"
				except Exception as e:
					print(f"Error occurred while calculating epsilon for delta={delta}: {e}")
					pass
			print('{}Train Epoch {}: [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(params.log_prefix,
			                                                                  epoch, n_processed, n_total, 100. * n_processed // n_total, loss.item()) + epsilon_str)

		if batch_idx + 1 == params.batches:
			break

	if prof:
		prof.stop()

	return losses


TestCriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], Any]
hamming_norm = lambda target, output: torch.norm(target - output, dim=1, p=1).mean().item()


def auroc_scores(targets: torch.Tensor, outputs: torch.Tensor) -> list[float]:
	from sklearn.metrics import roc_auc_score
	scores = []
	for j in range(targets.shape[1]):
		t = targets[:, j]
		valid = torch.isfinite(t) & ((t == 0) | (t == 1))
		t = t[valid]
		if t.numel() == 0 or t.min() == t.max():
			scores.append(float('nan'))
			continue
		scores.append(float(roc_auc_score(t.numpy(), outputs[valid, j].numpy())))
	return scores


def test(model: torch.nn.Module, params: Params, test_loader: DataLoader, criterion: TestCriterionType = hamming_norm):
	if params.skip_test:
		print("Skipping test")
		return 0.0

	print(f"Testing model on {len(test_loader)} points")
	start_time = time.time()

	fix_collate(test_loader)
	model.eval()

	accs = []
	all_targets = []
	all_outputs = []
	with torch.no_grad():
		for batch_idx, (inp, target) in enumerate(test_loader):
			inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
			output = model(inp)
			output = F.sigmoid(output)
			acc = criterion(target, output)
			accs.append(acc)
			all_targets.append(target.cpu())
			all_outputs.append(output.cpu())
			if batch_idx + 1 == params.batches:
				break

		elapsed_time = time.time() - start_time
		print(f"Testing took {elapsed_time:.2f}s")

		try:
			aurocs = auroc_scores(torch.cat(all_targets), torch.cat(all_outputs))
			valid = [a for a in aurocs if a == a]
			print(f"{params.log_prefix}Per-class AUROC: [{', '.join(f'{a:.6f}' for a in aurocs)}]")
			print(f"{params.log_prefix}Macro AUROC: {sum(valid) / len(valid):.6f}")
		except Exception as e:
			print(f"AUROC computation failed: {e}")

		return sum(accs) / len(accs)


save_name = "cxr"
save_model_path = "./checkpoints/xray_resnet/{name}_{phase}_{timestamp}.pt"
save_logs_path = "./logs/{name}_{phase}_{timestamp}.log"


def save(model=None, params=None, logs=None, model_path=None, logs_path=None, name=None):
	if model_path is None:
		model_path = save_model_path
	if logs_path is None:
		logs_path = save_logs_path
	if name is None:
		name = save_name

	if model is None:
		print("[save] model is None")

	if logs is None:
		print("[save] logs is None")

	timestamp = datetime.today().strftime('%m_%d_%H%M%S')

	if params is not None:
		if not isinstance(params, dict):
			params = params.__dict__
		props = {k: v for k, v in params.items() if not k.startswith("_")}.copy()

	if logs is not None:
		try:
			p = logs_path.format(phase=props['phase'], timestamp=timestamp, name=name)
			print(f"Saving logs to {p}")
			with open(p, "w+") as f:
				if logs is not None:
					props.update(
						epoch_time=logs['time'],
						loss_history=logs['loss'],
						acc_history=logs['acc']
					)
				f.write(json.dumps(props, default=str, indent='\t'))
		except Exception as e:
			print("exception occured when saving logs")
			print(props)
			print(e)

	if model is not None:
		try:
			p = model_path.format(phase=params['phase'] if params is not None else '', timestamp=timestamp, name=name)
			print(f"Saving model to {p}")
			torch.save(model.state_dict(), p)
		except Exception as e:
			print("exception occured when saving model")
			print(e)

	print("save() done")


save_model = lambda: None
save_logs = lambda: None


def run(model: torch.nn.Module, params: Params, train_dataloader: DataLoader, test_dataloader: DataLoader):
	print(params)
	loss_history = []
	time_history = []
	acc_history = []

	logs = {'loss': loss_history, 'time': time_history, 'acc': acc_history}
	global save_model, save_logs
	save_model = lambda: save(model=model, params=params, logs=None)
	save_logs = lambda: save(model=None, params=params, logs=logs)

	for epoch in range(0, params.epochs):
		print(f"Begin epoch {epoch}")
		start = time.time()

		loss_history.append(train(model, params, train_dataloader, epoch))

		epoch_time = time.time() - start
		time_history.append(epoch_time)
		print(f"Epoch {epoch} took {epoch_time:.2f}s")

		acc = test(model, params, test_dataloader)
		acc_history.append(acc)
		print(f"Acc:  {acc_history[-1]:.6f}")

		if params.save and epoch % 4 == 0 and epoch != params.epochs - 1:
			save(model=model, params=params)

	if params.save:
		save(model=model, params=params, logs=logs)
