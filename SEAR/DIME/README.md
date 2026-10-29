The Code for the DIME paper submission at ICML2025.
## DIME: Diffusion-Based Maximum Entropy Reinforcement Learning 

This repository accompanies the paper "[DIME: Diffusion-Based Maximum Entropy Reinforcement Learning](https://arxiv.org/pdf/2502.02316)" published at ICML 2025.

The file setup.sh provides a convenient way to set up the conda environment and install the required packages automatically via
```bash
chmod +x setup.sh
./setup.sh
```

After installation is finished, the conda environment can be activated, and the code can be run using 

```python
python run_dime.py
```

Specific parameters can be set in the terminal such as the learning environment using hydra's multirun function

```python
python run_dime.py --multirun env_name=dm_control/humanoid-run
```

Detailed hyperparameter specifications are available in the config directory. 

## Acknowledgements
Portions of the project are adapted from other repositories: 
- https://github.com/DenisBless/UnderdampedDiffusionBridges is licensed under MIT,
- https://github.com/adityab/CrossQ is licensed under MIT and is built upon code from "[Stable Baselines Jax](https://github.com/araffin/sbx/)"