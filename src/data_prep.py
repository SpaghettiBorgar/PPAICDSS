import argparse
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
from util.timer import Timer

chunk_dir = os.getenv("CHUNK_DIR", xray_data.DATA_DIR + '/chunks')
chunk_size = 256
resolution = 512

get_chunk_idx = lambda sample_idx: sample_idx // chunk_size
get_chunk_path = lambda chunk_idx: f"{chunk_dir}/chunk_{chunk_idx * chunk_size:06d}.pt.zstd"

num_chunks = int(math.ceil(xray_data.TOTAL_SAMPLES / chunk_size))


def make_chunk_transform(resolution=resolution):
	return v2.Compose([
		v2.Resize(size=None, max_size=resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
		v2.CenterCrop([resolution, resolution]),
		v2.ToDtype(torch.uint8),
		v2.ToImage(),
	])


chunk_transform = make_chunk_transform()
_worker_dataset: Dataset | None = None
_worker_chunk_dir: str | None = None
_worker_chunk_size: int | None = None
_worker_compression_level: int | None = None


def get_chunk(chunk_idx):
	with Timer(print=os.environ.get("DEBUG_CHUNK_LOADING", "0") == '1'):
		with zstd.open(get_chunk_path(chunk_idx), 'rb') as chunk_file:
			return torch.load(io.BytesIO(chunk_file.read()))


def generate_chunk(dataset: Dataset, file: str | IO, idx_start: int, idx_end: int, compression_level: int = 16, transform: Callable = lambda x: x):
	print(f"[{mp.current_process().name}] generating chunk {idx_start} - {idx_end}")
	if isinstance(file, str):
		chunk_file = zstd.open(file, mode='wb', level=compression_level)
	else:
		chunk_file = file
	torch.save(torch.stack([transform(dataset[i]) for i in range(idx_start, idx_end)]), f=chunk_file)
	if isinstance(file, str):
		chunk_file.close()


def _xray_chunk_transform(x):
	return x[0][0]


def _init_chunk_worker(resolution: int, torch_threads: int | None, chunk_dir: str, chunk_size: int, compression_level: int):
	global _worker_dataset, _worker_chunk_dir, _worker_chunk_size, _worker_compression_level
	if torch_threads is not None and torch_threads > 0:
		torch.set_num_threads(torch_threads)
	_worker_dataset = xray_data.XrayDataset(transform=make_chunk_transform(resolution), use_chunks=False)
	_worker_chunk_dir = chunk_dir
	_worker_chunk_size = chunk_size
	_worker_compression_level = compression_level


def _generate_chunk_by_index(chunk_idx: int):
	if _worker_dataset is None or _worker_chunk_dir is None or _worker_chunk_size is None or _worker_compression_level is None:
		raise RuntimeError("chunk worker was not initialized")
	idx_start = chunk_idx * _worker_chunk_size
	idx_end = min((chunk_idx + 1) * _worker_chunk_size, len(_worker_dataset))
	generate_chunk(
		_worker_dataset,
		f"{_worker_chunk_dir}/chunk_{idx_start:06d}.pt.zstd",
		idx_start,
		idx_end,
		_worker_compression_level,
		_xray_chunk_transform,
	)


def generate_chunks(
		chunk_dir=chunk_dir,
		chunk_size=chunk_size,
		resolution=resolution,
		compression_level=16,
		start_chunk=0,
		num_chunks_to_generate=None,
		num_workers=None,
		torch_threads=1,
):
	if chunk_size <= 0:
		raise ValueError("chunk_size must be positive")
	total_chunks = int(math.ceil(xray_data.TOTAL_SAMPLES / chunk_size))
	if start_chunk < 0:
		raise ValueError("start_chunk must be non-negative")
	if start_chunk >= total_chunks:
		raise ValueError(f"start_chunk must be less than {total_chunks}")

	if num_chunks_to_generate is None:
		stop_chunk = total_chunks
	else:
		if num_chunks_to_generate <= 0:
			raise ValueError("num_chunks_to_generate must be positive")
		stop_chunk = min(start_chunk + num_chunks_to_generate, total_chunks)

	if num_workers is None:
		num_workers = min(os.cpu_count() or 1, 6)
	if num_workers <= 0:
		raise ValueError("num_workers must be positive")
	if torch_threads is not None and torch_threads < 0:
		raise ValueError("torch_threads must be non-negative")

	print(f"Chunking data in {chunk_dir}")
	print(f"batch_size={chunk_size}, resolution={resolution}, workers={num_workers}")
	print(f"chunks={start_chunk}..{stop_chunk - 1} of {total_chunks}, torch_threads={torch_threads}")

	if not os.path.exists(chunk_dir):
		os.makedirs(chunk_dir)

	chunk_indices = range(start_chunk, stop_chunk)
	with mp.Pool(
			num_workers,
			initializer=_init_chunk_worker,
			initargs=(resolution, torch_threads, chunk_dir, chunk_size, compression_level),
	) as pool:
		pool.map(_generate_chunk_by_index, chunk_indices)


def parse_args():
	parser = argparse.ArgumentParser(description="Generate compressed xray image chunks")
	parser.add_argument("--chunk-dir", type=str, default=chunk_dir, help="Directory to write chunk files")
	parser.add_argument("--chunk-size", type=int, default=chunk_size, help="Samples per chunk")
	parser.add_argument("--resolution", "-R", type=int, default=resolution, help="Max side length to scale images to")
	parser.add_argument("--compression-level", "-C", type=int, default=16, help="zstd compression level")
	parser.add_argument("--workers", "-w", type=int, default=None, help="Worker processes to use")
	parser.add_argument("--num-chunks", "-N", type=int, default=None, help="Number of chunks to generate")
	parser.add_argument("--start-chunk", "-O", type=int, default=0, help="Chunk index to start at")
	parser.add_argument("--torch-threads", type=int, default=1, help="PyTorch native threads per worker process, or 0 to leave unchanged")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	generate_chunks(
		chunk_dir=args.chunk_dir,
		chunk_size=args.chunk_size,
		resolution=args.resolution,
		compression_level=args.compression_level,
		start_chunk=args.start_chunk,
		num_chunks_to_generate=args.num_chunks,
		num_workers=args.workers,
		torch_threads=args.torch_threads,
	)
