'''
CUSTOM ENVIRONMENT INIT :: AKSHAY
'''

import gymnasium as gym

from .custom_env_direct import UR5eRG2CustomTableEnvCfg, UR5eRG2CustomTableEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5eRG2-CustomDirect-v0",
    entry_point=f"{__name__}.custom_env_direct:UR5eRG2CustomTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_env_direct:UR5eRG2CustomTableEnvCfg",
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)