import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"

metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
annotations = pd.read_csv(f"{data_dir}/annotations.csv")


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
			img = self.transform(img)
		labels = torch.Tensor(sample.iloc[3:17].astype(float).values.copy()).to(torch.float32)
		view_label = metadata.loc[sample.dicom_id].ViewPosition
		_views = ['LATERAL', 'LL', 'PA', 'AP']
		view = view_label in _views and _views.index(view_label) or len(_views)
		view = F.one_hot(torch.tensor([view]).long(), num_classes=len(_views) + 1).squeeze()
		return (img, view), labels


class_weights = OrderedDict({'Atelectasis': 36574, 'Cardiomegaly': 34348, 'Consolidation': 7353, 'Edema': 19519, 'Enlarged Cardiomediastinum': 5317, 'Fracture': 3604, 'Lung Lesion': 6418, 'Lung Opacity': 43730, 'No Finding': 89070, 'Pleural Effusion': 41462, 'Pleural Other': 1748, 'Pneumonia': 14726, 'Pneumothorax': 7547, 'Support Devices': 42806})
total_samples = 221121
