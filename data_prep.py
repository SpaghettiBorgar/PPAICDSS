import sys

if sys.version_info >= (3, 14):
	from compression import zstd
else:
	from backports import zstd

import torchvision.transforms
from torchvision.transforms import v2

import xray_data
import torch
import os
from torch.utils.data import DataLoader

chunk_dir = xray_data.data_dir + '/chunks'
chunk_size = 1024
resolution = 600

get_chunk_idx = lambda sample_idx: sample_idx // chunk_size
get_chunk_path = lambda chunk_idx: f"{chunk_dir}/chunk_{chunk_idx * chunk_size:06d}.pt.zstd"

chunk_transform = v2.Compose([
	v2.Resize(size=None, max_size=resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
	v2.CenterCrop([resolution, resolution]),
	v2.ToImage(),
	v2.ToDtype(torch.int8)
])


def get_chunk(chunk_idx):
	with zstd.open(get_chunk_path(chunk_idx), 'rb') as chunk_file:
		return torch.load(chunk_file)


def generate_chunks(chunk_dir=chunk_dir, chunk_size=chunk_size, resolution=resolution):
	dataset = xray_data.XrayDataset(transform=chunk_transform, use_chunks=False)

	num_workers = min(os.cpu_count(), 8)
	dataloader = DataLoader(dataset, batch_size=chunk_size, num_workers=num_workers, in_order=True)

	print(f"Chunking data in {chunk_dir}")
	print(f"batch_size={chunk_size}, resolution={resolution}, workers={num_workers}")

	if not os.path.exists(chunk_dir):
		os.mkdir(chunk_dir)

	for batch_idx, batch in enumerate(dataloader):
		with zstd.open(f"{chunk_dir}/chunk_{batch_idx * chunk_size:06d}.pt.zstd", mode='wb', level=16) as chunk_file:
			print(f"Writing to {chunk_file.name}")
			torch.save(batch, f=chunk_file)


if __name__ == '__main__':
	generate_chunks()
