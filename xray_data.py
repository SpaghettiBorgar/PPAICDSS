import os
from collections import OrderedDict
from multiprocessing import shared_memory, current_process

import numpy as np
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

chunk_cache_size = 4
chunk_cache = OrderedDict()


class XrayDataset(Dataset):
	def __init__(self, img_dir=img_root, cache_index=None, offset=0, size=0, transform=None, use_chunks=True):
		self.img_dir = img_dir
		self.transform = transform
		self.size = size if size > 0 else (len(annotations.index) - offset + size if offset >= 0 else -offset)
		self.offset = offset if offset >= 0 else len(annotations.index) + offset
		self.use_chunks = use_chunks
		self.cache_index = cache_index
		self.cur_chunk_idx = None

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		index += self.offset
		sample = annotations.loc[index]

		if self.use_chunks:
			chunk_idx = index // data_prep.chunk_size
			if not self.cur_chunk_idx == chunk_idx:
				if self.cache_index is not None:
					index_lock = self.cache_index['lock']
					index_lock.acquire()
					if chunk_idx in self.cache_index:
						index_lock.release()
						while not 'shm_name' in self.cache_index[chunk_idx]:
							os.sched_yield()
						with index_lock:
							meta = self.cache_index[chunk_idx]
						shm = shared_memory.SharedMemory(name=meta['shm_name'])
						chunk_arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
					else:
						self.cache_index[chunk_idx] = {}
						index_lock.release()
						chunk_tensor = data_prep.get_chunk(chunk_idx)[0][0]
						arr = chunk_tensor.numpy()
						shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
						chunk_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
						chunk_arr[:] = arr[:]
						with index_lock:
							self.cache_index[chunk_idx] = {
								'shm_name': shm.name,
								'shape': arr.shape,
								'dtype': str(arr.dtype)
							}
					self.cur_chunk_idx, self.cur_chunk, self.cur_shm = chunk_idx, torch.from_numpy(chunk_arr), shm
				else:
					if not chunk_idx in chunk_cache:
						if len(chunk_cache) >= chunk_cache_size:
							chunk_cache.popitem(last=False)
						chunk_cache.update({chunk_idx: data_prep.get_chunk(chunk_idx)[0][0]})

					chunk_cache.move_to_end(chunk_idx)
					self.cur_chunk_idx, self.cur_chunk = chunk_idx, chunk_cache[chunk_idx]

			img = self.cur_chunk[index % data_prep.chunk_size]
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
