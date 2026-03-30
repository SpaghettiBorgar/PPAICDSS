import json
import os
import signal
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode, v2

print("Initializing")

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"
device = "cuda"

metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
annotations = pd.read_csv(f"{data_dir}/annotations.csv")

training_data_size = len(annotations.index)

initial_weights = models.DenseNet121_Weights.IMAGENET1K_V1
# checkpoint = torch.load("./checkpoints/cnn_03_29_222512.pt", weights_only=True)
checkpoint = None


class XrayModel(nn.Module):
	def __init__(self, num_labels=14, xray_view_dim=0):
		super().__init__()

		self.xray_view_dim = xray_view_dim

		self.densenet = models.densenet121(weights=initial_weights)
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

		if checkpoint is not None:
			self.load_state_dict(checkpoint)

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

start_index = 0
batchsize = 256
epoch_size = 10
epochs = 10
lr = 1e-3
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(cnn.parameters(), lr=lr, weight_decay=1e-3)
resolution = 224
transform = v2.Compose([
	v2.Resize(size=None, max_size=resolution, interpolation=InterpolationMode.BICUBIC),
	v2.CenterCrop([resolution, resolution]),
	# v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	v2.Normalize(mean=[(0.485 + 0.456 + 0.406) / 3], std=[(0.229 + 0.224 + 0.225) / 3])
])


# plt.imshow(F.pad(cnn.parameters()[0].detach(), (0,1,0,1), value=1.0).reshape(8,8,3,8,8).permute(0,3,1,4,2).flatten(start_dim=0,end_dim=1).flatten(start_dim=1,end_dim=2))

def train(img, view, target):
	cnn.train()
	optimizer.zero_grad()
	output = cnn(img, xray_view=view)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	print(f"{loss.item():.6f}")
	return loss.item()


def test(img, view, target):
	cnn.eval()
	with torch.no_grad():
		output = cnn(img, xray_view=view)
		output = F.sigmoid(output)
		acc = torch.norm(target - output, dim=1, p=1)
		return torch.mean(acc).item()


def get_sample(n):
	sample = annotations.loc[n]
	img = decode_image(img_root + '/' + sample.image_file).to(device, torch.float32) / 256
	img = transform(img)
	labels = torch.Tensor(sample.iloc[3:17].astype(float).values.copy()).to(device, torch.float32)
	view_label = metadata.loc[sample.dicom_id].ViewPosition
	VIEWS = ['LATERAL', 'LL', 'PA', 'AP']
	view = view_label in VIEWS and VIEWS.index(view_label) or 4
	view = F.one_hot(torch.tensor([view], device=device).long(), num_classes=5).squeeze()
	return img, view, labels


def get_batch(offset, size=batchsize):
	[imgs, views, labels] = list(zip(*(get_sample(n) for n in range(offset, offset + size))))
	return torch.stack(imgs), torch.stack(views), torch.stack(labels)


def save():
	timestamp = datetime.today().strftime('%m_%d_%H%M%S')
	print(f"saving to {timestamp}")
	with open(f"./logs/train_{timestamp}.log", "w+") as f:
		params = {
			'weights': str(initial_weights),
			'resolution': resolution,
			'batchsize': batchsize,
			'epochs': epochs,
			'criterion': str(criterion),
			'optimizer': str(optimizer),
			'epoch_time': sum(time_history) / len(time_history),
			'loss_history': loss_history,
		}
		f.write(json.dumps(params, indent='\t'))

	torch.save(cnn.state_dict(), f"./checkpoints/cnn_{timestamp}.pt")


signal.signal(signal.SIGUSR1, lambda sig, frame: save())
signal.signal(signal.SIGTERM, lambda sig, frame: save())

for param in cnn.densenet.parameters():
	param.requires_grad = False

loss_history = []
time_history = []
acc_history = []
for epoch in range(0, epochs):
	print(f"epoch {epoch}")
	start = time.time()
	losses = list(
		train(*get_batch(start_index + batchsize * (batch_idx + epoch * epoch_size), batchsize)) for batch_idx in
		range(epoch_size))
	loss_history.append(sum(losses) / len(losses))
	time_history.append(time.time() - start)
	print(f"loss: {loss_history[-1]:.6f}")
	accs = list(test(*get_batch((training_data_size - batchsize * (batch_idx + 1)))) for batch_idx in range(4))
	acc_history.append(sum(accs) / len(accs))
	print(f"acc:  {acc_history[-1]:.6f}")

save()
