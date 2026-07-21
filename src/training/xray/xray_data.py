import logging
import multiprocessing as mp
import os
import shutil
import time
from collections import OrderedDict
from multiprocessing import shared_memory, current_process
from multiprocessing.managers import SharedMemoryManager, DictProxy, SyncManager
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

DATA_DIR = os.getenv("TRAIN_DATA_DIR", default="./data")
IMG_ROOT = f"{DATA_DIR}/images"

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
          "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
          "Pneumonia", "Pneumothorax", "Support Devices"]
LABELS_SHORT = ["Atelect", "Cardiomeg", "Consolid", "Edema", "Enl.Cardm", "Fracture",
                "Lung Lesn", "Lung Opac", "No Find", "Pleur.Eff", "Pleur.Oth", "Pneumonia", "PneumoTrx",
                "Supp. Dev"]

CLASS_WEIGHTS = OrderedDict(
	{'Atelectasis': 36574, 'Cardiomegaly': 34348, 'Consolidation': 7353, 'Edema': 19519,
	 'Enlarged Cardiomediastinum': 5317, 'Fracture': 3604, 'Lung Lesion': 6418,
	 'Lung Opacity': 43730, 'No Finding': 89070, 'Pleural Effusion': 41462, 'Pleural Other': 1748,
	 'Pneumonia': 14726, 'Pneumothorax': 7547, 'Support Devices': 42806})
DATASET_TOTAL_SAMPLES = 221121
CLASS_POS_WEIGHTS = DATASET_TOTAL_SAMPLES / torch.tensor(list(CLASS_WEIGHTS.values())) - 1
TOTAL_SAMPLES = int(os.getenv("TRAIN_DATA_COUNT", default=221121))
TEST_OFFSET = - min(20000, TOTAL_SAMPLES // 10)
TEST_SIZE = int(os.environ.get("TEST_SIZE", "0"))
TRAIN_SIZE = TOTAL_SAMPLES + TEST_OFFSET
TRAIN_SIZE &= ~1  # avoid single-batch edge case

DATA_MEAN = 0.4010714861347213
DATA_STD = 0.10697800868384252

import data_prep

metadata = pd.read_csv(f"{DATA_DIR}/metadata.csv").set_index('dicom_id')
annotations = pd.read_csv(f"{DATA_DIR}/annotations.csv")

mp_man: SyncManager | None = None
smm: SharedMemoryManager | None = None
shared_index: DictProxy | None = None

_sample_labels: torch.Tensor | None = None
_sample_views: torch.Tensor | None = None

logger = logging.getLogger(__name__)


def _ensure_sample_tensors():
	global _sample_labels, _sample_views
	if _sample_labels is not None:
		return
	_sample_labels = torch.tensor(annotations.iloc[:, 3:17].to_numpy(dtype=np.float32))
	_views = ['LATERAL', 'LL', 'PA', 'AP']
	view_pos = metadata.ViewPosition.reindex(annotations.dicom_id)
	view_idx = torch.tensor([_views.index(v) if v in _views else len(_views) for v in view_pos], dtype=torch.long)
	_sample_views = F.one_hot(view_idx, num_classes=len(_views) + 1)


def split_transform(transform):
	"""Split a Compose at the first ToDtype: everything before it keeps chunks uint8
	for the shm cache; ToDtype/Normalize run per sample."""
	if isinstance(transform, v2.Compose):
		ts = list(transform.transforms)
		for i, t in enumerate(ts):
			if isinstance(t, v2.ToDtype):
				return (v2.Compose(ts[:i]) if i > 0 else None), v2.Compose(ts[i:])
	return transform, None


def setup_shm():
	global mp_man, smm, shared_index
	mp_man = mp.Manager()
	smm = SharedMemoryManager()
	smm.start()
	shared_index = mp_man.dict()
	shared_index['lock'] = mp_man.Lock()
	shared_index['bytes'] = 0


def shutdown_shm():
	global mp_man, smm, shared_index
	try:
		smm.shutdown()
		mp_man.shutdown()
	except AttributeError:
		pass


class XrayDataset(Dataset):
	classes = LABELS

	def __init__(self, img_dir=IMG_ROOT, cache_index=None, shm_manager=None, offset=0, size=0, transform=None, use_chunks=True):
		self.img_dir = img_dir
		self.transform = transform
		self.size = size if size > 0 else (TOTAL_SAMPLES - offset + size if offset >= 0 else -offset)
		self.offset = offset if offset >= 0 else TOTAL_SAMPLES + offset
		if self.offset < 0 or self.offset + self.size > TOTAL_SAMPLES:
			raise ValueError(f"invalid offset/size: {offset}/{size} for dataset of {TOTAL_SAMPLES} samples")
		self.use_chunks = use_chunks
		self.cache_index = cache_index if cache_index is not None else shared_index
		self.smm = shm_manager if shm_manager is not None else smm
		assert (self.cache_index is None) == (self.smm is None)
		self._transform_tag = None
		self._chunk_stage = None  # transform prefix applied per chunk (stays uint8, cached)
		self._sample_stage = None  # ToDtype/Normalize tail applied per sample at load time
		self._chunk_tag = None
		self._chunks = {}  # worker-local: chunk_idx -> view into a shm-cached chunk (uint8)
		self._shms = {}  # keeps attached SharedMemory objects alive for those views
		self._local_chunks = OrderedDict()  # LRU for chunks that couldn't go to shm; these own their memory
		self._shm_full_warned = False
		_ensure_sample_tensors()

	def __len__(self):
		return self.size

	def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
		index = self.real_index(index)
		if _sample_labels is None:
			# __init__ builds these pre-fork, but under spawn/forkserver (Python 3.14 default)
			# the dataset arrives pickled and module globals start out None
			_ensure_sample_tensors()

		if self.use_chunks:
			img = self.get_ready_chunk(index // data_prep.chunk_size)[index % data_prep.chunk_size]
			if self._sample_stage:
				img = self._sample_stage(img)
		else:
			img = decode_image(self.img_dir + '/' + annotations.image_file.iloc[index])
			if self.transform:
				img = self.transform(img)

		return (img, _sample_views[index]), _sample_labels[index]

	def get_ready_chunk(self, chunk_idx):
		tag = repr(self.transform)
		if tag != self._transform_tag:  # transform was (re)assigned, e.g. between FL phases
			self._transform_tag = tag
			self._chunk_stage, self._sample_stage = split_transform(self.transform)
			self._chunk_tag = repr(self._chunk_stage)  # cache key: phases differing only in the sample stage share chunks
			self._chunks.clear()
			self._shms.clear()
			self._local_chunks.clear()
		chunk = self._chunks.get(chunk_idx)
		if chunk is None and chunk_idx in self._local_chunks:
			self._local_chunks.move_to_end(chunk_idx)
			chunk = self._local_chunks[chunk_idx]
		if chunk is None:
			chunk, shared = self._load_ready_chunk(self._chunk_tag, chunk_idx)
			if shared:  # shm views are cheap, keep them all; local copies own RAM, so LRU-cap them
				self._chunks[chunk_idx] = chunk
			else:
				self._local_chunks[chunk_idx] = chunk
				while len(self._local_chunks) > int(os.getenv("LOCAL_CHUNK_CACHE", 4)):
					self._local_chunks.popitem(last=False)
		return chunk

	def _prepare_chunk(self, chunk_idx):
		chunk = data_prep.get_chunk(chunk_idx)
		return self._chunk_stage(chunk) if self._chunk_stage else chunk

	def _load_ready_chunk(self, tag, chunk_idx):
		if self.cache_index is None:
			return self._prepare_chunk(chunk_idx), False

		key = (tag, chunk_idx)
		lock = self.cache_index['lock']
		with lock:
			meta = self.cache_index.get(key)
			if meta is None:
				self.cache_index[key] = {}  # claim; others wait below until shm_name appears

		if meta is None:
			arr = self._prepare_chunk(chunk_idx).contiguous().numpy()
			shm = self._shm_alloc(arr.nbytes)
			if shm is None:  # tmpfs nearly full — a write into an overcommitted segment would SIGBUS
				with lock:
					del self.cache_index[key]  # release the claim so waiters fall back too
				return torch.from_numpy(arr), False
			chunk_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
			chunk_arr[:] = arr[:]
			with lock:
				self.cache_index[key] = {
					'shm_name': shm.name,
					'shape': arr.shape,
					'dtype': str(arr.dtype)
				}
				self.cache_index['bytes'] = self.cache_index['bytes'] + arr.nbytes
			if os.environ.get("DEBUG_CHUNK_LOADING", "0") == '1':
				logger.debug(f"[{current_process().name}] cached chunk {chunk_idx} ({len(self.cache_index) - 2}/{data_prep.num_chunks})")
		else:
			while meta is not None and not 'shm_name' in meta:
				time.sleep(0.005)
				meta = self.cache_index.get(key)
			if meta is None:  # creator gave up (shm full); keep a worker-local copy instead
				return self._prepare_chunk(chunk_idx), False
			shm = shared_memory.SharedMemory(name=meta['shm_name'], track=False)  # SharedMemoryManager owns it
			chunk_arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)

		self._shms[chunk_idx] = shm
		return torch.from_numpy(chunk_arr), True

	def _shm_alloc(self, nbytes):
		limit = float(os.getenv("XRAY_SHM_MAX_GB", "inf")) * (1 << 30)
		if self.cache_index['bytes'] + nbytes > limit:
			reason = "XRAY_SHM_MAX_GB reached"
		else:
			try:
				free = shutil.disk_usage('/dev/shm').free
				with open('/proc/meminfo') as f:  # tmpfs pages need physical RAM; a fault under pressure SIGBUSes
					for line in f:
						if line.startswith('MemAvailable:'):
							free = min(free, int(line.split()[1]) * 1024)
							break
			except OSError:
				free = None
			if free is None or free >= nbytes + (2 << 30):
				return self.smm.SharedMemory(size=nbytes)
			reason = f"{free >> 20} MiB headroom left"
		if not self._shm_full_warned:
			self._shm_full_warned = True
			logger.warning(f"[{current_process().name}] shm chunk cache exhausted ({reason}), keeping further chunks worker-local")
		return None

	def real_index(self, idx):
		return idx + self.offset
