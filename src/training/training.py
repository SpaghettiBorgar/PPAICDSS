import json
import time
from datetime import datetime
from typing import TypeAlias, Callable, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from training.params import Params
from util.mapping import tree_map


def tuple_preserving_collate(batch):
	return _lists_to_tuples(default_collate(batch))


def _lists_to_tuples(x):
	if isinstance(x, list):
		return tuple(_lists_to_tuples(v) for v in x)
	if isinstance(x, dict):
		return {k: _lists_to_tuples(v) for k, v in x.items()}
	return x


def fix_collate(data_loader: DataLoader):
	if data_loader.collate_fn == default_collate:
		data_loader.collate_fn = tuple_preserving_collate


def train(model: torch.nn.Module, params: Params, train_loader: DataLoader, epoch):
	fix_collate(train_loader)
	model.train()
	losses = []

	optimizer = params.get_optimizer()
	criterion = params.get_criterion()

	for batch_idx, (inp, target) in enumerate(train_loader):
		inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
		optimizer.zero_grad()
		output = model(inp)
		loss = criterion(output, target)

		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		if batch_idx % (1 if params.phase == 'testing' else 4) == 0:
			n_total = len(train_loader.dataset) if params.batches == 0 else min(len(train_loader.dataset), params.batches * params.batch_size)
			n_processed = min((batch_idx + 1) * params.batch_size, n_total)
			print('Train Epoch {}: [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
				epoch, n_processed, n_total, 100. * n_processed // n_total, loss.item()))
		if batch_idx + 1 == params.batches:
			break

	return losses


TestCriterionType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], Any]
hamming_norm = lambda target, output: torch.norm(target - output, dim=1, p=1).mean().item()


def test(model: torch.nn.Module, params: Params, test_loader: DataLoader, criterion: TestCriterionType = hamming_norm):
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


def save(model=None, params=None, logs=None, model_path=save_model_path, logs_path=save_logs_path):
	timestamp = datetime.today().strftime('%m_%d_%H%M%S')

	if logs is not None:
		try:
			p = logs_path.format(phase=params.phase, timestamp=timestamp)
			print(f"Saving logs to {p}")
			with open(p, "w+") as f:
				o = vars(params).copy()
				if logs is not None:
					o.update(
						epoch_time=logs['time'],
						loss_history=logs['loss'],
						acc_history=logs['acc']
					)
				f.write(json.dumps(o, default=str, indent='\t'))
		except Exception as e:
			print(e)

	if model is not None:
		try:
			p = model_path.format(phase=params.phase if params is not None else '', timestamp=timestamp)
			print(f"Saving model to {p}")
			torch.save(model.state_dict(), p)
		except Exception as e:
			print(e)


do_save = lambda: None


def run(model: torch.nn.Module, params: Params, train_dataloader: DataLoader, test_dataloader: DataLoader):
	print(params)
	loss_history = []
	time_history = []
	acc_history = []

	logs = {'loss': loss_history, 'time': time_history, 'acc': acc_history}
	global do_save
	do_save = lambda: save(model=model, params=params, logs=logs)

	for epoch in range(0, params.epochs):
		print(f"Begin epoch {epoch}")
		start = time.time()

		loss_history += train(model, params, train_dataloader, epoch)

		epoch_time = time.time() - start
		time_history.append(epoch_time)
		print(f"Epoch {epoch} took {epoch_time:.2f}s")

		acc = test(model, params, test_dataloader)
		acc_history.append(acc)
		print(f"Acc:  {acc_history[-1]:.6f}")

		if params.save and epoch % 1 == 0 and epoch != params.epochs - 1:
			save(model=model, params=params)

	if params.save:
		do_save()
