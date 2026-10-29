# CQN-AS as used in SEARs experiments

This folder contains a fork of the official CQN-AS codebase that we used to generate the 
CQN-AS experiments in the SEAR paper. 

The most interesting files for reproducing our results are `train_cqn_as_metaworld.py` and `metaworld_src/config_cqn_as_metaworld.yaml`.

The rest of this readme is mostly the original readme with some additional instructions for using uv.



# Coarse-to-fine Q-Network with Action Sequence

This repository provides the re-implementation of Coarse-to-fine Q-Network (**CQN**) and Coarse-to-fine Q-Network with Action Sequence (**CQN-AS**), introduced in:

[**Continuous Control with Coarse-to-fine Reinforcement Learning**](https://younggyo.me/cqn/)\
[Younggyo Seo](https://younggyo.me/), [Jafar Uruç](https://github.com/JafarAbdi), [Stephen James](https://stepjam.github.io/)\
CoRL 2024

[**Reinforcement Learning with Action Sequence for Data-Efficient Robot Learning**](https://younggyo.me/cqn-as/)\
[Younggyo Seo](https://younggyo.me/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)\
Preprint

In summary, **CQN** is a sample-efficient value-based RL algorithm that uses *discrete actions* for solving continuous control problems.
The key idea is to apply multiple levels of discretization to continuous action space, and train RL agents to zoom-into continuous action space in a coarse-to-fine manner.

**CQN-AS** extends **CQN** by training a critic network that outputs *Q-values over a series of actions*, allowing for learning useful value functions from noisy training data such as human-collected demonstrations.

We also provide implementation of **DrQ-v2+**, a variant of DrQ-v2 that is highly optimized for demo-driven RL setup.
![gif1](media/cqn_gif1.gif)
![gif2](media/cqn_gif2.gif)

## Quick Setup (Recommended)

The easiest way to set up this repository on any machine with identical dependencies:

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Dependencies
```bash
git clone <your-repo-url>
cd CQN-AS

# Install from frozen requirements (ensures exact versions)
uv venv
uv pip install -r requirements-frozen.txt
```

**Note**: The standard `uv sync` doesn't work due to a gymnasium version conflict between dependencies. Use the frozen requirements instead.

### 3. MuJoCo Setup (Required for MetaWorld)
```bash
# Download and install MuJoCo 2.1.0
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

# Add to your shell rc file (~/.bashrc or ~/.zshrc)
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc
```

### 4. Test the Setup
```bash
# Test MetaWorld (minimal dependencies)
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  uv run --no-sync python train_cqn_as_metaworld.py \
  task_name=pick-place-v3 \
  num_train_frames=2000 \
  num_seed_frames=600 \
  eval_every_frames=50000 \
  save_video=false \
  save_train_video=false \
  batch_size=8 \
  replay_buffer_num_workers=0
```

### Alternative: Manual Installation
If you don't want to use frozen requirements:
```bash
uv venv

# Install core dependencies
uv pip install torch tensordict==0.6.0 packaging==24.2 termcolor==2.4.0 \
  imageio==2.34.1 imageio_ffmpeg==0.5.1 hydra-core==1.1.0 \
  hydra-submitit-launcher==1.1.5 opencv-python-headless==4.10.0.82 \
  numpy==1.26.4 tensorboard==2.17.0 dm_env==1.6 dm_control==1.0.19 \
  wandb==0.17.4 mujoco==3.1.5

# Install gymnasium and metaworld
uv pip install "gymnasium>=1.1" "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@master"

# Install BiGym (requires SSH or HTTPS access)
uv pip install "bigym @ git+ssh://git@github.com/chernyadev/bigym.git@72d305437d5a13800ea633479a1060619fc14e54"
# OR: uv pip install "bigym @ git+https://github.com/chernyadev/bigym.git@72d305437d5a13800ea633479a1060619fc14e54"
```

## Updates
- We provide logs used for drawing learning curves in `logs` directory.
Note that this can be different from the curves in the paper (at the point of Jan 27th, 2025). We will update the paper soon.

```
import pickle
# BENCHMARK = {humanoidbench, rlbench, bigym}
with open("BENCHMARK_results.pkl", "rb") as f:
    logs = pickle.load(f)
```

## Misc
- We adopt a domain-based code maintenance design to avoid having a too complex code design. In other words, instead of having a shared code that can be used for all domains, we maintain a separate set of files for each domain.
- This codebase might not fully reproduce the experiments in the paper due to potential human errors in porting code. Please let us know if you encounter any discrepancy in the performance between the reported results and the results from running this code.

## TO-DO List
- [ ] Refactor to use `ReplayBuffer`, `LazyTensorStorage` of `torchrl`
- [ ] Refactor to replace numpy-based TemporalEnsemble implementation with torch-based implementation
- [ ] Check if the code fully reproduces the results in the paper.

## Experimental Setups
Our codebase supports running CQN and CQN-AS on various domains popular for researchers. We plan to support more domains and any PR or suggestion to support more domains (including yours) is welcome!

| Domain | State-based | Pixel-based | Action Sequence | Demo-driven | Stable |
|-|-|-|-|-|-|
| [BiGym](https://github.com/chernyadev/bigym) | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: |:white_check_mark:|
| [RLBench](https://github.com/stepjam/RLBench) | :x: | :white_check_mark:| :white_check_mark: | :white_check_mark: |:white_check_mark:|
| [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) | :white_check_mark: | :x: | :white_check_mark: | :x: |:white_check_mark:|
| [MetaWorld](https://github.com/Farama-Foundation/Metaworld) | :white_check_mark: | :x: | :white_check_mark: | :x: |:white_check_mark:|
| [DMC](https://github.com/google-deepmind/dm_control) | :white_check_mark: | :white_check_mark: | :x: | :x: |:warning:|


## Installation
Install conda environment:
```
conda env create -f conda_env.yml
conda activate cqn
```

For faster experiments, install the nightly version of PyTorch, and set `use_compile` to True.
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

## Instructions for BiGym experiments

Install BiGym (latest version at the date of Oct 29th should be used).

```
git clone https://github.com/chernyadev/bigym.git
cd bigym
git checkout 72d305437d5a13800ea633479a1060619fc14e54
pip install -e .
```

BiGym will download demonstrations and save pre-processed cache in the first run of each task, so there is no need for each user to handle demonstrations by themselves.
Our code also automatically handles demonstrations that can't be replayed by saving them as training data but not demonstration data.

Run experiments (CQN and CQN-AS):
```
# Run CQN
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train_cqn_bigym.py bigym_task=move_plate seed=1

# Run CQN-AS (Ours)
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train_cqn_as_bigym.py bigym_task=move_plate seed=1

# Run DrQ-v2+
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train_drqv2plus_bigym.py bigym_task=move_plate seed=1
```

## Instructions for RLBench experiments

Install RLBench and PyRep (latest versions at the date of Oct 29th, 2024 should be used).
Follow the guide in original repositories for (1) installing RLBench and PyRep and (2) enabling headless mode. (See README in [RLBench](https://github.com/stepjam/RLBench) \& [Robobase](https://github.com/robobase-org/robobase?tab=readme-ov-file#rlbench) for information on installing RLBench.)

```
git clone https://github.com/stepjam/RLBench
git clone https://github.com/stepjam/PyRep
# Install PyRep
cd PyRep
git checkout 8f420be8064b1970aae18a9cfbc978dfb15747ef
pip install .
# Install RLBench
cd RLBench
git checkout b80e51feb3694d9959cb8c0408cd385001b01382
pip install .
```

Pre-collect demonstrations
```
cd RLBench/rlbench
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python dataset_generator.py --save_path=/your/own/directory --image_size 84 84 --renderer opengl3 --episodes_per_task 100 --variations 1 --processes 1 --tasks take_lid_off_saucepan --arm_max_velocity 2.0 --arm_max_acceleration 8.0
```

Run experiments :
```
# Run CQN
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python train_cqn_rlbench.py rlbench_task=take_lid_off_saucepan num_demos=100 dataset_root=/your/own/directory

# Run CQN-AS (Ours)
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python train_cqn_as_rlbench.py rlbench_task=take_lid_off_saucepan num_demos=100 dataset_root=/your/own/directory

# Run DrQ-v2+
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python train_drqv2plus_rlbench.py rlbench_task=take_lid_off_saucepan num_demos=100 dataset_root=/your/own/directory
```

## Instructions for HumanoidBench experiments

Install HumanoidBench
```
git clone https://github.com/carlosferrazza/humanoid-bench.git
cd humanoid-bench
pip install -e .
```

**Note**: HumanoidBench requires `opencv-python==4.10.0.84` to be installed but this can cause RLBench experiments to fail. In that case, uninstall `opencv-python==4.10.0.84` and install `opencv-python-headless==4.10.0.84`. You can run both experiments with a single virtual environment with headless opencv.


Run experiments:
```
# Run CQN
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train_cqn_humanoid.py task_name=h1hand-stand-v0

# Run CQN-AS
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train_cqn_as_humanoid.py task_name=h1hand-stand-v0
```

Note that `h1hand` is different from `h1` and `h1hand` is the default task used in the original benchmark.

## Instructions for MetaWorld experiments

MetaWorld v3 is automatically installed via the setup instructions above. It provides 50 robotic manipulation tasks.

Run experiments:
```bash
# Run CQN
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync python train_cqn_metaworld.py task_name=pick-place-v3 seed=1

# Run CQN-AS
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 uv run --no-sync python train_cqn_as_metaworld.py task_name=pick-place-v3 seed=1
```

Available tasks (v3): `pick-place-v3`, `door-open-v3`, `drawer-open-v3`, `button-press-v3`, `window-open-v3`, and 45 more. See [MetaWorld documentation](https://github.com/Farama-Foundation/Metaworld) for the full list.

**Note**: MetaWorld uses gymnasium 1.2+ and requires MuJoCo 2.1.0 bindings (see Quick Setup section above). Task names ending in `-v2` are automatically converted to `-v3`.

## Instructions for DMC experiments

Run experiments:
```
# For pixel-based experiments
CUDA_VISIBLE_DEVICES=0 python train_cqn_dmc.py dmc_task=cartpole_swingup

# For state-based experiments
CUDA_VISIBLE_DEVICES=0 python train_cqn_dmc_state.py dmc_task=cartpole_swingup
```

**Warning**: CQN is not extensively tested in DMC


## Acknowledgements
This repository is based on public implementation of [DrQ-v2](https://github.com/facebookresearch/drqv2)



## Citation
```
# CQN
@inproceedings{seo2024continuous,
  title={Continuous Control with Coarse-to-fine Reinforcement Learning},
  author={Seo, Younggyo and Uru{\c{c}}, Jafar and James, Stephen},
  booktitle={Conference on Robot Learning},
  year={2024}
}

# CQN-AS
@article{seo2024reinforcement,
  title={Reinforcement Learning with Action Sequence for Data-Efficient Robot Learning},
  author={Seo, Younggyo and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2411.12155},
  year={2024}
}
```
