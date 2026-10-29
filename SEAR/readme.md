# SEAR code submission

To use this code, you need a linux PC with [uv](https://astral.sh/uv) installed and a Nvidia GPU.
Uv takes care of venv management and installs all dependencies the first time you run any of the commands below.

```bash
# SEAR results 
uv run run.py \
	wandb.activate=true \
	env_name=metaworld-push-v1 \
	alg.tot_time_steps=1000000 \
	alg.critic_chunksize=10 \
	alg.actor_chunksize=10

# SimbaV2 results
uv run run.py \
	wandb.activate=true \
	env_name=metaworld-push-v1 \
	alg.tot_time_steps=1000000 \
	alg.critic_chunksize=1 \
	alg.actor_chunksize=1
    actor.kind=mlp \
    critic.hs="[512,512]" \
    critic.kind=mlp \
    critic.simbav2=true \
    optimizer.critic_use_adamw=false

# Naive Chunking results
uv run run.py \
	wandb.activate=true \
	env_name=metaworld-push-v1 \
	alg.tot_time_steps=1000000 \
	alg.critic_chunksize=10 \
	alg.actor_chunksize=10 \
    actor.kind=mlp \
    critic.hs="[512,512]" \
    critic.kind=mlp \
    critic.simbav2=true \
    critic.only_fit_last_nstep=true \
    optimizer.critic_use_adamw=false \
    random_chunks=false 
```
The hyperparameters as used in our experiments can be found in `configs/sac.yaml`.

# Acknowledgements

This code is based on [CrossQ](https://github.com/adityab/CrossQ/tree/main)'s codebase which is based on [sbx](https://github.com/araffin/sbx) which is based on [stablebaselines3](https://github.com/DLR-RM/stable-baselines3).

The code under `scale_rl/` is a vendored version of the official SimbaV2 release. We use its hyperspherical normalized MLPs.
The code under `DIME/` is a vendored version of the official DIME release. We use it to implement a diffusion actor which is not discussed in the paper.
