import numpy as np

VEC_LEN = 8
KEY_LEN = 4
NUM_PEERS = 5

# global modulus prime number
q = 71

"""
Goal of the SMPC scheme is to divide the peer space into small groups, of which the server can construct the sums of
 their private keys of, which can be used to decrypt only the sum of those peers' previously received cyphertexts.
The scheme allows to securely compute only the sums without revealing any information that would make individual keys or
 plaintexts reconstructable.
In groups of k peers, each peer splits their key into k+1 shares. One gets sent to the server, one is kept private, and
 the rest is distributed to the other peers.
Each peer will then have received a share of each other peer. Combined with the privately-held share, the sum computed
 by each peer contains one fragment of each peer's key, that the server collects and finally adds the remaining share to
 in order to construct the total key sum.
For groups of k, an individual peer's key or plaintext is only reconstructable if ALL other k-1 peers AND the server are
 colluding.
"""

# secret vectors
rng = np.random.default_rng(seed=0)
x = [rng.integers(0, 10, size=VEC_LEN) for _ in range(NUM_PEERS)]
print(f"Vectors:\n{'  '.join([f"{i}: {x[i]}" for i in range(NUM_PEERS)])}")

# private keys
rng = np.random.default_rng(seed=1)
s = [rng.integers(0, q, size=KEY_LEN) for _ in range(NUM_PEERS)]
print(f"Keys:\n{'  '.join([f"{i}: {s[i]}" for i in range(NUM_PEERS)])}")

# encrypted vectors
rng = np.random.default_rng(seed=2)
c = [x_i + s_i % q for x_i, s_i in zip(x, map(lambda arr: np.resize(arr, VEC_LEN), s))]
print(f"Cyphertexts:\n{'  '.join([f"{i}: {c[i]}" for i in range(NUM_PEERS)])}")


def compose_keys(*peers):
	# each peer generates a random share for each other peer plus itself
	shares = {i: {j: rng.integers(0, q, size=KEY_LEN) for j in peers} for i in peers}

	print(f"Shares (row sends to column):")
	print(f"\t" + '\t\t'.join([str(p) for p in [*peers, 'S']]))

	# last share gets calculated by subtracting the other shares from the key
	# this gets sent to the server
	for i, shares_i in shares.items():
		shares_i['server'] = ((s[i] - sum(shares_i.values())) % q)
		assert all(sum(shares_i.values()) % q == s[i])
		print(f"{i}\t" + '\t'.join([str(shares_i[j]) for j in [*peers, 'server']]))

	# sum of shares as received by each peer
	shares_sums = {i: sum([shares[j][i] for j in peers]) % q for i in peers}

	return shares_sums, {i: shares[i]['server'] for i in peers}


# --- server side ---

print("\nFirst group with peers 0, 1")

c_sum_1 = sum(c[0:2]) % q
print(f"Cypher sum 1:\t{c_sum_1}")

shares_p2p_sums, shares_s = compose_keys(0, 1)
# construct sum of group's private keys by adding all received shares
key_1 = (sum(shares_p2p_sums.values()) + sum(shares_s.values())) % q
print(f"\nKey 1:\t\t {key_1}")
assert all(key_1 == sum(s[0:2]) % q)

x_sum_1 = (c_sum_1 - np.resize(key_1, VEC_LEN)) % q
print(f"Decrypted sum 1: {x_sum_1}")

print("\nSecond group with peers 2, 3, 4")

c_sum_2 = sum(c[2:5]) % q
print(f"Cypher sum 2:\t{c_sum_1}")

shares_p2p_sums, shares_s = compose_keys(2, 3, 4)
key_2 = (sum(shares_p2p_sums.values()) + sum(shares_s.values())) % q
print(f"\nKey 2:\t\t {key_2}")
assert (key_2 == sum(s[2:5]) % q).all()

x_sum_2 = (c_sum_2 - np.resize(key_2, VEC_LEN)) % q
print(f"Decrypted sum 2: {x_sum_2}")

x_sum = x_sum_1 + x_sum_2
print("\nActual vector sum:")
print(sum(x))
print("\nDecrypted vector sum:")
print(x_sum)
