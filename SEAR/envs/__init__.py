from gymnasium.envs.registration import register

from envs.Slit_Env_v2 import BaseEnv
from envs.GoalReaching import BaseEnv as GoalBaseEnv
from envs.TestEnvs import Final100, Const1

register(
    id="SlitEnv-v2",
    entry_point="envs.Slit_Env_v2:SlitEnv",
    max_episode_steps=BaseEnv._HORIZON,
    kwargs={
        "seed": 0,
        "obs_keys": "state",
        "render_mode": None,
        "fixed_start_goal": True,
    },
)

register(
    id="SlitEnv-v3",
    entry_point="envs.Slit_Env_v2:SlitEnv",
    max_episode_steps=BaseEnv._HORIZON,
    kwargs={
        "seed": 0,
        "obs_keys": "state",
        "render_mode": None,
        "fixed_start_goal": True,
        "slit_width": 0.9,
    },
)

register(
    id="GoalReach-v0",
    entry_point="envs.GoalReaching:BaseEnv",
    max_episode_steps=GoalBaseEnv._HORIZON,
    kwargs={
        "seed": 0,
        "obs_keys": "state",
        "render_mode": None,
        "fixed_start_goal": False,
    },
)
register(
    id="GoalReach-v1",
    entry_point="envs.GoalReaching:BaseEnv",
    max_episode_steps=GoalBaseEnv._HORIZON,
    kwargs={
        "seed": 0,
        "obs_keys": "state",
        "render_mode": None,
        "fixed_start_goal": True,
    },
)

register(
    id="GoalReachSparse-v0",
    entry_point="envs.GoalReaching:BaseEnv",
    max_episode_steps=GoalBaseEnv._HORIZON,
    kwargs={
        "seed": 0,
        "obs_keys": "state",
        "render_mode": None,
        "fixed_start_goal": False,
        "sparse": True,
    },
)

register(
    id="Final100-v0",
    entry_point="envs.TestEnvs:Final100",
    max_episode_steps=10,
    kwargs={
        "seed": 0,
    },
)

register(
    id="Const1-v0",
    entry_point="envs.TestEnvs:Const1",
    max_episode_steps=10,
    kwargs={
        "seed": 0,
    },
)
