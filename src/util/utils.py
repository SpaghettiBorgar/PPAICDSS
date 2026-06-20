from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from typing import Any, Iterator, Union, Sequence

import numpy as np


def auto_type(val):
	from ast import literal_eval
	try:
		return literal_eval(val)
	except:
		return val


def make_seed(seed_input: Any) -> Union[int, Sequence[int]]:
	if isinstance(seed_input, int) or (
			isinstance(seed_input, (list, tuple, np.ndarray)) and all(isinstance(x, int) for x in seed_input)):
		return seed_input

	if isinstance(seed_input, (bytes, bytearray)):
		input_bytes = seed_input
	elif isinstance(seed_input, str):
		input_bytes = seed_input.encode('utf-8')
	else:
		try:
			input_bytes = bytes(seed_input)
		except TypeError as e:
			raise TypeError(f"Input type {type(seed_input)} is not supported.") from e

	hash_obj = hashlib.sha256(input_bytes)
	hash_bytes = hash_obj.digest()

	# Convert to sequence of 32-bit unsigned integers (4 bytes per int)
	seed_sequence = [
		int.from_bytes(hash_bytes[i:i + 4], byteorder='big', signed=False)
		for i in range(0, len(hash_bytes), 4)
	]

	return seed_sequence


def random_partitions(
		xs: range,
		parts: int,
		seed: int | None = None,
		evenness: float = 1.0,
) -> Iterable[range]:
	"""
	Partition a range into `parts` contiguous random subranges.

	Parameters
	----------
	xs:
		Input range to partition.
	parts:
		Number of partitions to create.
	seed:
		Optional random seed.
	evenness:
		Controls how balanced the partition sizes are.

		- 1.0  -> as even as possible
		- 0.0  -> completely random
		- values in between smoothly interpolate

	Returns
	-------
	Iterable[range]
		Contiguous subranges whose union equals `xs`.

	Notes
	-----
	- Partitions are contiguous.
	- Empty partitions are possible if `parts > len(xs)`.
	- Works with arbitrary range steps.
	"""

	if parts <= 0:
		raise ValueError("parts must be > 0")

	if not (0.0 <= evenness <= 1.0):
		raise ValueError("evenness must be between 0.0 and 1.0")

	rng = random.Random(seed)

	n = len(xs)

	# Trivial case
	if parts == 1:
		yield xs
		return

	# Base equal distribution
	base = n // parts
	remainder = n % parts

	ideal_sizes = [
		base + (1 if i < remainder else 0)
		for i in range(parts)
	]

	# Maximum deviation budget
	# higher randomness => more variance
	randomness = 1.0 - evenness

	if randomness == 0:
		sizes = ideal_sizes
	else:
		# Generate random weights
		weights = [rng.random() for _ in range(parts)]
		total = sum(weights)

		random_sizes = [
			int(round(n * w / total))
			for w in weights
		]

		# Blend equal and random distributions
		blended = [
			round(
				evenness * ideal + randomness * random_size
			)
			for ideal, random_size in zip(ideal_sizes, random_sizes)
		]

		# Final correction
		drift = n - sum(blended)

		while drift != 0:
			i = rng.randrange(parts)

			if drift > 0:
				blended[i] += 1
				drift -= 1
			elif blended[i] > 0:
				blended[i] -= 1
				drift += 1

		sizes = blended

	# Emit subranges
	start = xs.start
	step = xs.step

	for size in sizes:
		stop = start + size * step
		yield range(start, stop, step)
		start = stop


def chunk_with_min_remainder(seq, n, n_min=1):
	seq = list(seq)
	if n <= 0 or n_min <= 0 or n_min > n:
		raise ValueError("Require n > 0, 0 < n_min <= n")

	chunks = [seq[i:i + n] for i in range(0, len(seq), n)]
	if len(chunks) <= 1:
		return chunks

	# If the last chunk is too small, steal from previous chunks (preserving order)
	need = n_min - len(chunks[-1])
	i = len(chunks) - 2
	while need > 0 and i >= 0:
		can_take = max(0, len(chunks[i]) - n_min)  # don't shrink donor below n_min
		take = min(need, can_take)
		if take:
			moved = chunks[i][-take:]
			chunks[i] = chunks[i][:-take]
			chunks[i + 1] = moved + chunks[i + 1]
			need -= take
		else:
			i -= 1

	if need > 0:
		raise ValueError("Cannot satisfy n_min without making a chunk smaller than n_min.")

	return chunks

def resilient_iter(it: Iterator):
	while True:
		try:
			yield next(it)
		except StopIteration:
			return
		except Exception as e:
			print(f"Skipping item due to error: {e}")
			continue