import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import visualize
from util.mapping import tree_map

from torch.utils.data._utils.collate import default_collate

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

def train(model: torch.nn.Module, params, train_loader: DataLoader, epoch, batches=0):
	model.train()
	losses = []

	for batch_idx, (inp, target) in enumerate(train_loader):
		inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
		params.optimizer.zero_grad()
		output = model(inp)
		loss = params.criterion(output, target)

		loss.backward()
		params.optimizer.step()
		losses.append(loss.item())
		if batch_idx % (1 if params.phase == 'testing' else 4) == 0:
			n_total = len(train_loader.dataset) if batches == 0 else min(len(train_loader.dataset), batches * params.batchsize)
			n_processed = min((batch_idx + 1) * params.batchsize, n_total)
			print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, n_processed, n_total, 100. * n_processed // n_total, loss.item()))
		if batch_idx + 1 == batches:
			break

	return losses


def test(model: torch.nn.Module, params, test_loader: DataLoader, batches=0):
	model.eval()

	accs = []
	with torch.no_grad():
		for batch_idx, (inp, target) in enumerate(test_loader):
			inp, target = tree_map(lambda t: t.to(params.device), (inp, target))
			output = model(inp)
			output = F.sigmoid(output)
			acc = torch.norm(target - output, dim=1, p=1).mean()
			accs.append(acc.item())
			if batch_idx + 1 == batches:
				break

		return sum(accs) / len(accs)


def save(model=None, params=None, logs=None, name_suffix=""):
	timestamp = datetime.today().strftime('%m_%d_%H%M%S')
	name = timestamp + ('_' + name_suffix if len(name_suffix) > 0 else '')
	print(f"Saving as {name}")

	if params is not None:
		with open(f"./logs/train_{name}.log", "w+") as f:
			o = vars(params).copy()
			if logs is not None:
				o.update(
					epoch_time=sum(logs['time']) / max(len(logs['time']), 1),
					loss_history=logs['loss'],
					acc_history=logs['acc']
				)
			f.write(json.dumps(o, default=str, indent='\t'))

	if model is not None:
		torch.save(model.state_dict(), f"./checkpoints/xray_{name}.pt")


do_save = lambda: None


def run(model: torch.nn.Module, params, train_dataloader: DataLoader, test_dataloader: DataLoader):
	print(params)
	loss_history = []
	time_history = []
	acc_history = []

	fix_collate(train_dataloader)
	fix_collate(test_dataloader)

	logs = {'loss': loss_history, 'time': time_history, 'acc': acc_history}
	global do_save
	do_save = lambda: save(model=model, params=params, logs=logs, name_suffix=str(params.phase))

	for epoch in range(0, params.epochs):
		print(f"Begin epoch {epoch}")
		start = time.time()

		loss_history += train(model, params, train_dataloader, epoch, params.batches)

		epoch_time = time.time() - start
		time_history.append(epoch_time)
		print(f"Epoch {epoch} took {epoch_time:.2f}s")

		acc = test(model, params, test_dataloader, 2)
		acc_history.append(acc)
		print(f"Acc:  {acc_history[-1]:.6f}")

		if params.save and epoch % 1 == 0 and epoch != params.epochs - 1:
			save(model, name_suffix=str(params.phase))

	if params.save:
		do_save()
