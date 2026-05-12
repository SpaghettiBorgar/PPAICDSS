import io
import math
import multiprocessing as mp
import sys
from typing import IO, Callable

if sys.version_info >= (3, 14):
	from compression import zstd
else:
	from backports import zstd

import torchvision.transforms
from torchvision.transforms import v2

from training.xray import xray_data
import torch
import os
from torch.utils.data import Dataset

chunk_dir = xray_data.DATA_DIR + '/chunks'
chunk_size = 128
resolution = 600

get_chunk_idx = lambda sample_idx: sample_idx // chunk_size
get_chunk_path = lambda chunk_idx: f"{chunk_dir}/chunk_{chunk_idx * chunk_size:06d}.pt.zstd"

num_chunks = int(math.ceil(xray_data.TOTAL_SAMPLES / chunk_size))

chunk_transform = v2.Compose([
	v2.Resize(size=None, max_size=resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
	v2.CenterCrop([resolution, resolution]),
	v2.ToDtype(torch.uint8),
	v2.ToImage(),
])


def get_chunk(chunk_idx):
	with zstd.open(get_chunk_path(chunk_idx), 'rb') as chunk_file:
		return torch.load(io.BytesIO(chunk_file.read()))


def generate_chunk(dataset: Dataset, file: str | IO, idx_start: int, idx_end: int, compression_level: int = 16, transform: Callable = lambda x: x):
	print(f"[{mp.current_process().name}] generating chunk {idx_start} - {idx_end}")
	if isinstance(file, str):
		chunk_file = zstd.open(file, mode='wb', level=compression_level)
	torch.save(torch.stack([transform(dataset[i]) for i in range(idx_start, idx_end)]), f=chunk_file)
	if isinstance(file, str):
		chunk_file.close()


def _xray_chunk_transform(x):
	return x[0][0]


def generate_chunks(chunk_dir=chunk_dir, chunk_size=chunk_size, resolution=resolution, compression_level=16):
	dataset = xray_data.XrayDataset(transform=chunk_transform, use_chunks=False)

	num_workers = min(os.cpu_count(), 16)

	print(f"Chunking data in {chunk_dir}")
	print(f"batch_size={chunk_size}, resolution={resolution}, workers={num_workers}")

	if not os.path.exists(chunk_dir):
		os.mkdir(chunk_dir)

	with mp.Pool(num_workers) as pool:
		pool.starmap(generate_chunk, ((
			dataset,
			f"{chunk_dir}/chunk_{i * chunk_size:06d}.pt.zstd",
			i * chunk_size,
			min((i + 1) * chunk_size, len(dataset)),
			compression_level,
			_xray_chunk_transform)
			for i in range(num_chunks)))


if __name__ == '__main__':
	generate_chunks()
