import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image

data_dir = os.getenv("TRAIN_DATA_DIR", default="./data")
img_root = f"{data_dir}/images"

import data_prep

metadata = pd.read_csv(f"{data_dir}/metadata.csv").set_index('dicom_id')
annotations = pd.read_csv(f"{data_dir}/annotations.csv")

chunk_cache_size = 3


class XrayDataset(Dataset):
	def __init__(self, img_dir=img_root, offset=0, size=0, transform=None, use_chunks=True):
		self.img_dir = img_dir
		self.transform = transform
		self.size = size if size > 0 else (len(annotations.index) - offset + size if offset >= 0 else -offset)
		self.offset = offset if offset >= 0 else len(annotations.index) + offset
		self.use_chunks = use_chunks

		self.chunk_cache = OrderedDict()

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		index += self.offset
		sample = annotations.loc[index]

		if self.use_chunks:
			chunk_idx = index // data_prep.chunk_size
			if chunk_idx not in self.chunk_cache:
				print(f"loading chunk {chunk_idx}")
				if len(self.chunk_cache) >= chunk_cache_size:
					self.chunk_cache.popitem(last=False)
				self.chunk_cache.update({chunk_idx: data_prep.get_chunk(chunk_idx)})

			self.chunk_cache.move_to_end(chunk_idx)
			chunk = self.chunk_cache[chunk_idx]

			img = chunk[0][0][index % data_prep.chunk_size]
		else:
			img = decode_image(self.img_dir + '/' + sample.image_file)

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
