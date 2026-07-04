import json
import time
from datetime import datetime
from typing import TypeAlias, Callable, Any

from opacus.validators import ModuleValidator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.params import Params
from util.dp_compat import make_model_dp_compatible, fix_inplace
from util.mapping import tree_map
from util.utils import resilient_iter, fix_collate


def train(model: torch.nn.Module, params: Params, train_loader: DataLoader, epoch):
	fix_collate(train_loader)
	model.train()
	losses = []

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
		if batch_idx % (1 if params.phase == 'testing' else 10) == 0:
			n_total = len(train_loader.dataset) if params.batches == 0 else min(len(train_loader.dataset), params.batches * params.batch_size)
			n_processed = min((batch_idx + 1) * params.batch_size, n_total)
			epsilon_str = ""
			if params.target_delta is not None:
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

	return losses


TestCriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], Any]
hamming_norm = lambda target, output: torch.norm(target - output, dim=1, p=1).mean().item()


def test(model: torch.nn.Module, params: Params, test_loader: DataLoader, criterion: TestCriterionType = hamming_norm):
	print(f"Testing model on {len(test_loader)} points")
	fix_collate(test_loader)
	model.eval()

	accs = []
	with torch.no_grad():
		for batch_idx, (inp, target) in enumerate(test_loader):
			inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
			output = model(inp)
			output = F.sigmoid(output)
			acc = criterion(target, output)
			accs.append(acc)
			if batch_idx + 1 == params.batches:
				break

		return sum(accs) / len(accs)


save_model_path = "./checkpoints/xray_resnet/{phase}_{timestamp}.pt"
save_logs_path = "./logs/xray_{phase}_{timestamp}.log"


def save(model=None, params=None, logs=None, model_path=None, logs_path=None):
	if model_path is None:
		model_path = save_model_path
	if logs_path is None:
		logs_path = save_logs_path
	
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
			p = logs_path.format(phase=props['phase'], timestamp=timestamp)
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
			p = model_path.format(phase=params['phase'] if params is not None else '', timestamp=timestamp)
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

		if params.save and epoch % 5 == 0 and epoch != params.epochs - 1:
			save(model=model, params=params)

	if params.save:
		save(model=model, params=params, logs=logs)
0
