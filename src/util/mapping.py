from collections.abc import Mapping


def tree_map(fn, obj):
	if isinstance(obj, tuple):
		return tuple(tree_map(fn, x) for x in obj)

	elif isinstance(obj, list):
		return [tree_map(fn, x) for x in obj]

	elif isinstance(obj, Mapping):
		return {k: tree_map(fn, v) for k, v in obj.items()}

	else:
		return fn(obj)
