# Code Repository

## Setup

1. [Install uv](https://docs.astral.sh/uv/#installation).
2. Set up `.env`
3. `source .env`
4. `uv pip install -e .`

## Entry points

Make sure to always `source .env`
Following scripts can then be run via `./run`:

| Script | Purpose |
|---|---|
| `src/training/train.py` | Centralized training. `-p <phase>` preset, `-c <checkpoint>`, `-P key=value` overrides. |
| `src/sim/federated_learning.py` | FL simulation. `-n` participants, `-p` phase(s), `--use-smpc`. |
| `src/eval/xray_test.py` | CXR CNN accuracy evaluation. |
| `src/eval/mia.py` | Loss-based membership inference attack. |
| `src/eval/gradinversion.py` | Gradient inversion attack. |
| `src/data_prep.py` | Dataset pre-processing and chunking |
| `src/visualize` | Visualization of training logs and more |

Use `--help` on any script for details

Enabling DP: pass either `-P grad_norm=X -P noise_mult=Y`, or
`-P target_epsilon=X -P target_delta=Y -P grad_norm=Z`.
