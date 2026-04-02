import json
import time
from datetime import datetime

import torch
import torch.nn.functional as F


def train(model, params, train_loader, epoch, batches=0):
	model.train()
	losses = []

	for batch_idx, ((img, view), target) in enumerate(train_loader):
		img, view, target = img.to(params.device), view.to(params.device), target.to(params.device)
		params.optimizer.zero_grad()
		output = model(img, xray_view=view)
		loss = params.criterion(output, target)
		loss.backward()
		params.optimizer.step()
		losses.append(loss.item())
		if batch_idx % 1 == 0:
			num_samples = len(train_loader.dataset) if batches == 0 else min(len(train_loader.dataset), batches * train_loader.batch_size)
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, (batch_idx + 1) * len(img), num_samples,
					   100. * (batch_idx * params.batchsize) / num_samples, loss.item()))
		if batch_idx + 1 == batches:
			break

	return losses


def test(model, params, test_loader, batches=0):
	model.eval()

	accs = []
	with torch.no_grad():
		for batch_idx, ((img, view), target) in enumerate(test_loader):
			img, view, target = img.to(params.device), view.to(params.device), target.to(params.device)
			output = model(img, xray_view=view)
			output = F.sigmoid(output)
			acc = torch.norm(target - output, dim=1, p=1).mean()
			accs.append(acc.item())
			if batch_idx + 1 == batches:
				break

		return sum(accs) / len(accs)


def save(model=None, params=None, logs=None, name_suffix=""):
	timestamp = datetime.today().strftime('%m_%d_%H%M%S')
	name = timestamp + ('_' + name_suffix if len(name_suffix) > 0 else '')
	print(f"saving as {name}")

	if params is not None:
		with open(f"./logs/train_{name}.log", "w+") as f:
			o = vars(params).copy()
			if logs is not None:
				o.update(
					epoch_time=sum(logs['time']) / min(len(logs['time']), 1),
					loss_history=logs['loss'],
					acc_history=logs['acc']
				)
			f.write(json.dumps(o, default=str, indent='\t'))

	if model is not None:
		torch.save(model.state_dict(), f"./checkpoints/xray_{name}.pt")


do_save = lambda: None


def run(model, params, train_dataloader, test_dataloader):
	print(params)
	loss_history = []
	time_history = []
	acc_history = []

	logs = {'loss': loss_history, 'time': time_history, 'acc': acc_history}
	global do_save
	do_save = lambda: save(model=model, params=params, logs=logs, name_suffix=params.phase)

	for epoch in range(0, params.epochs):
		print(f"Epoch {epoch}")
		start = time.time()
		loss_history += train(model, params, train_dataloader, epoch, params.batches)
		time_history.append(time.time() - start)

		acc = test(model, params, test_dataloader, 2)
		acc_history.append(acc)
		print(f"Acc:  {acc_history[-1]:.6f}")

		if epoch % 1 == 0 and epoch != params.epochs - 1:
			save(model, name_suffix=params.phase)

	do_save()
