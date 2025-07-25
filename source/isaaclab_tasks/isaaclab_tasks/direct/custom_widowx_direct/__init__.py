'''
CUSTOM ENVIRONMENT INIT :: AKSHAY
'''

import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Isaac-WidowX250-CustomDirect-v0",
    entry_point=f"{__name__}.widowx_env_direct:WidowX250CustomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.widowx_env_direct:WidowX250CustomEnvCfg",
    },
)