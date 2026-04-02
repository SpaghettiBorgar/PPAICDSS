import json
import os
import signal
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode, v2

import data_analysis

print("Initializing")

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"
checkpoints_dir = "./checkpoints"
device = "cuda"

metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
annotations = pd.read_csv(f"{data_dir}/annotations.csv")

training_data_size = len(annotations.index)

phase1 = dict(
	batchsize=512,
	batches=80,
	epochs=3,
	resolution=384,
	lr=1e-3,
	weight_decay=1e-3,
	freeze_backend=True,
	checkpoint=None
)
phase2 = dict(
	batchsize=128,
	batches=256,
	epochs=10,
	resolution=384,
	lr=1e-4,
	weight_decay=1e-3,
	freeze_backend=False,
	checkpoint="cnn_04_01_135749.pt"
)
phase3 = dict(
	batchsize=64,
	batches=64,
	epochs=4,
	resolution=600,
	lr=1e-4,
	weight_decay=1e-4,
	freeze_backend=False,
	checkpoint="cnn_04_02_021828.pt"
)
params = SimpleNamespace(**phase1)


class XrayModel(nn.Module):
	def __init__(self, num_labels=14, xray_view_dim=0):
		super().__init__()

		self.xray_view_dim = xray_view_dim

		self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
		old_weights = self.densenet.features.conv0.weight.data
		self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.densenet.features.conv0.weight.data = old_weights.mean(dim=1, keepdim=True)

		num_features = self.densenet.classifier.in_features
		self.densenet.classifier = nn.Identity()
		self.classifier = nn.Linear(num_features + xray_view_dim, num_labels)

		self.metanet = nn.Sequential(
			nn.Linear(xray_view_dim, 32),
			nn.ReLU()
		)

		self.classifier = nn.Sequential(
			nn.Linear(32 + num_features, 512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.25),

			nn.Linear(512, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(0.25),

			nn.Linear(256, num_labels)
		)

		if params.checkpoint is not None:
			self.load_state_dict(torch.load(f"{checkpoints_dir}/{params.checkpoint}", weights_only=True))

	def forward(self, x, xray_view=None):
		x = self.densenet(x)

		if xray_view is None:
			xray_view = torch.zeros((x.shape[0], self.xray_view_dim))
		xray_view = xray_view.to(torch.float32)

		xray_view = self.metanet(xray_view)
		x = torch.cat([x, xray_view], dim=1)

		x = self.classifier(x)

		return x


cnn = XrayModel(num_labels=14, xray_view_dim=5).to(device)

class_weights = data_analysis.total_samples / torch.tensor(list(data_analysis.class_weights.values())) - 1
class_weights = class_weights.to(device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = torch.optim.AdamW(cnn.parameters(), lr=params.lr, weight_decay=params.weight_decay)
transform = v2.Compose([
	v2.Resize(size=None, max_size=params.resolution, interpolation=InterpolationMode.BICUBIC),
	v2.CenterCrop([params.resolution, params.resolution]),
	v2.ToImage(),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[(0.485 + 0.456 + 0.406) / 3], std=[(0.229 + 0.224 + 0.225) / 3])
])


# plt.imshow(F.pad(cnn.parameters()[0].detach(), (0,1,0,1), value=1.0).reshape(8,8,3,8,8).permute(0,3,1,4,2).flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2))

class XrayDataset(Dataset):
	def __init__(self, img_dir, offset=0, size=0, transform=None):
		self.img_dir = img_dir
		self.transform = transform
		self.size = size if size > 0 else (len(annotations.index) - offset + size if offset >= 0 else -offset)
		self.offset = offset if offset >= 0 else len(annotations.index) + offset

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		index += self.offset
		sample = annotations.loc[index]
		img = decode_image(img_root + '/' + sample.image_file)
		if self.transform:
			img = transform(img)
		labels = torch.Tensor(sample.iloc[3:17].astype(float).values.copy()).to(torch.float32)
		view_label = metadata.loc[sample.dicom_id].ViewPosition
		_views = ['LATERAL', 'LL', 'PA', 'AP']
		view = view_label in _views and _views.index(view_label) or len(_views)
		view = F.one_hot(torch.tensor([view]).long(), num_classes=len(_views) + 1).squeeze()
		return (img, view), labels


def train(train_loader, epoch, batches=0):
	cnn.train()
	losses = []

	for batch_idx, ((img, view), target) in enumerate(train_loader):
		img, view, target = img.to(device), view.to(device), target.to(device)
		optimizer.zero_grad()
		output = cnn(img, xray_view=view)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		if batch_idx % 1 == 0:
			num_samples = len(train_loader.dataset) if batches == 0 else min(len(train_loader.dataset), batches * train_loader.batch_size)
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, (batch_idx + 1) * len(img), num_samples,
					   100. * (batch_idx * params.batchsize) / num_samples, loss.item()))
		if batch_idx + 1 == batches:
			break

	return losses


def test(test_loader, epoch, batches=0):
	cnn.eval()

	accs = []
	with torch.no_grad():
		for batch_idx, ((img, view), target) in enumerate(test_loader):
			img, view, target = img.to(device), view.to(device), target.to(device)
			output = cnn(img, xray_view=view)
			output = F.sigmoid(output)
			acc = torch.norm(target - output, dim=1, p=1).mean()
			accs.append(acc.item())
			if batch_idx + 1 == batches:
				break

		return sum(accs) / len(accs)


def save(save_logs=True, save_model=True):
	timestamp = datetime.today().strftime('%m_%d_%H%M%S')
	print(f"saving as {timestamp}")

	if save_logs:
		with open(f"./logs/train_{timestamp}.log", "w+") as f:
			o = vars(params).copy()
			o.update(
				criterion=criterion,
				optimizer=optimizer,
				epoch_time=sum(time_history) / len(time_history),
				loss_history=loss_history,
				acc_history=acc_history
			)
			f.write(json.dumps(o, default=str, indent='\t'))

	if save_model:
		torch.save(cnn.state_dict(), f"./checkpoints/cnn_{timestamp}.pt")


signal.signal(signal.SIGUSR1, lambda sig, frame: save())
signal.signal(signal.SIGTERM, lambda sig, frame: save() or sys.exit(0))

num_workers = min(os.cpu_count(), 16)
training_data = XrayDataset(img_root, offset=0, size=200000, transform=transform)
testing_data = XrayDataset(img_root, offset=-20000, size=2048, transform=transform)
train_dataloader = DataLoader(training_data, batch_size=params.batchsize, num_workers=num_workers, pin_memory=True, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=params.batchsize, num_workers=num_workers, pin_memory=True, shuffle=True)

if params.freeze_backend:
	for param in cnn.densenet.parameters():
		param.requires_grad = False

loss_history = []
time_history = []
acc_history = []

if __name__ == '__main__':
	print(params)
	for epoch in range(0, params.epochs):
		print(f"Epoch {epoch}")
		start = time.time()
		loss_history = train(train_dataloader, epoch, params.batches)
		time_history.append(time.time() - start)
		acc = test(testing_dataloader, epoch)
		acc_history.append(acc)
		print(f"acc:  {acc_history[-1]:.6f}")
		save(save_logs=False)

	save()
