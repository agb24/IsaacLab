'''
CUSTOM ENVIRONMENT INIT :: AKSHAY
'''

import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5eRG2-CustomDirect-v1",
    entry_point=f"{__name__}.custom_ik_env_direct:UR5eRG2TableIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_ik_env_direct:UR5eRG2CustomTableEnvCfg",
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)