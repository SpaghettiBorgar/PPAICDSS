from __future__ import annotations

import random
from collections.abc import Iterable


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
