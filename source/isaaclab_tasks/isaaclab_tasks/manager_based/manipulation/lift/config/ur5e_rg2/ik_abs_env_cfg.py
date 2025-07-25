# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#-----------> PLACE INTO <ISAACLAB_PATH>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur5e_rg2

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots_ur5e_rg2 import UR5E_HIGH_PD_CFG  # isort: skip

##
# Rigid object lift environment.
##

@configclass
class UR5eCubeLiftEnvCfg(joint_pos_env_cfg.UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set UR5e as robot
        self.scene.robot = UR5E_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR5e)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
            body_name="gripper_center",

            # joint_names=["panda_joint.*"],
            # body_name="panda_hand",

            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1]),
        )

@configclass
class UR5eCubeLiftEnvCfg_PLAY(UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

##
# Deformable object lift environment.
##

@configclass
class UR5eTeddyBearLiftEnvCfg(UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )

        # Make the end effector less stiff to not hurt the poor teddy bear
        self.scene.robot.actuators["onrobot_rg2_base_link"].effort_limit = 50.0
        self.scene.robot.actuators["onrobot_rg2_base_link"].stiffness = 40.0
        self.scene.robot.actuators["onrobot_rg2_base_link"].damping = 10.0

        # # Make the end effector less stiff to not hurt the poor teddy bear
        # self.scene.robot.actuators["panda_hand"].effort_limit = 50.0
        # self.scene.robot.actuators["panda_hand"].stiffness = 40.0
        # self.scene.robot.actuators["panda_hand"].damping = 10.0

        # Disable replicate physics as it doesn't work for deformable objects
        self.scene.replicate_physics = False

        # Set events for the specific object type (deformable cube)
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Remove all the terms for the state machine demo
        self.terminations.object_dropping = None
        self.rewards.reaching_object = None
        self.rewards.lifting_object = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.observations.policy.object_position = None