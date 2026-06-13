import numpy as np

KEY_LEN = 2048
DTYPE = np.uint32
QUANT_DTYPE = np.int16
QUANT_MIN = - 2 ** 15
QUANT_MAX = + 2 ** 15
MOD = 2147483647

rng = np.random.default_rng(0)


def generate_key_mask(rng=rng):
	return rng.integers(low=0, high=MOD, size=KEY_LEN)


def make_last_share(key, shares):
	return (key - sum(shares)) % MOD


def encrypt(vec, key):
	return (vec + np.resize(key, len(vec))) % MOD


def decrypt(vec, key):
	return (vec - np.resize(key, len(vec))) % MOD


def normalize(vec):
	norm = np.linalg.norm(vec)
	if norm == 0:
		raise ValueError("Trying to normalize zero-vector")
	return vec / norm


def quantize(vec):
	return np.round(np.clip(vec * QUANT_MAX, QUANT_MIN, QUANT_MAX)).astype(QUANT_DTYPE)


def unquantize(vec):
	return vec.astype(np.float32) / QUANT_MAX


def recover_sign(vec):
	half = (MOD - 1) // 2
	return np.where(vec > half, vec - MOD, vec)
