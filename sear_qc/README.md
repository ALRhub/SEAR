# SEAR-QC Code

This repository contains the code to reproduce the QC-SEAR results. It extends the official [Q-Chunking (QC)](https://github.com/colinqiyangli/qc) implementation, on which this codebase is directly based.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), a Python package manager. Install it via:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install project dependencies:

```bash
uv sync
```

No manual virtual environment setup is required — `uv` handles this automatically.

## Reproducing Results

The commands below reproduce one run of each method on a single task. The full experiments sweep over tasks 1–4 and seeds 42–49.

**Our method** (SEAR-QC, MLP critic):
```bash
MUJOCO_GL=egl uv run python main.py \
    --seed=42 \
    --run_group=reproduce \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=cube-triple-play-singletask-task1-v0 \
    --sparse=False \
    --horizon_length=5 \
    --agent.critic_backbone=mlp 
```

**Baseline** (QC):
```bash
git checkout og_qc
MUJOCO_GL=egl uv run python main.py \
    --seed=42 \
    --run_group=reproduce \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=cube-triple-play-singletask-task1-v0 \
    --sparse=False \
    --horizon_length=5
```

## Acknowledgments

This codebase is built on top of the official [Q-Chunking](https://github.com/colinqiyangli/qc) implementation, which in turn builds on [FQL](https://github.com/seohongpark/fql). The `rlpd_networks/` and `rlpd_distributions/` folders are taken from [RLPD](https://github.com/ikostrikov/rlpd).
