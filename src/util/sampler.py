import random

from torch.utils.data import Sampler


class BlockShuffleBatchSampler(Sampler):
	def __init__(self, dataset_size, chunk_size, block_size, batch_size, drop_last=False):
		self.dataset_size = dataset_size
		self.chunk_size = chunk_size
		self.block_size = block_size
		self.batch_size = batch_size
		self.drop_last = drop_last

		self.num_chunks = dataset_size // chunk_size
		self.blocks_per_chunk = chunk_size // block_size

	def __iter__(self):
		blocks = []

		for chunk_start in range(0, self.dataset_size, self.chunk_size):
			chunk_end = min(chunk_start + self.chunk_size, self.dataset_size)

			for block_start in range(chunk_start, chunk_end, self.block_size):
				block_end = min(block_start + self.block_size, chunk_end)
				block = list(range(block_start, block_end))
				random.shuffle(block)  # shuffle within blocks locally
				blocks.append(block)

		# shuffle blocks globally
		random.shuffle(blocks)

		# emit full batches
		batch = []
		for block in blocks:
			batch.extend(block)

			while len(batch) >= self.batch_size:
				yield batch[:self.batch_size]
				batch = batch[self.batch_size:]

		if batch and not self.drop_last:
			yield batch

	def __len__(self):
		if self.drop_last:
			return self.dataset_size // self.batch_size
		else:
			return (self.dataset_size + self.batch_size - 1) // self.batch_size
