# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#-----------> PLACE INTO <ISAACLAB_PATH>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur5e_rg2

import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

# ##
# # Joint Position Control
# ##

# gym.register(
#     id="Isaac-Lift-Cube-Franka-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Lift-Cube-Franka-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
#     disable_env_checker=True,
# )

# ##
# # Inverse Kinematics - Absolute Pose Control
# ##

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-UR5e-RG2-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR5eCubeLiftEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
