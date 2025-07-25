# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots_ur10e_gripper import UR10_CFG  # isort: skip


@configclass
class UR10CubeLiftEnvCfg(joint_pos_env_cfg.UR10CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set UR10 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR10)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"],
            body_name="robotiq_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1]),
        )

        # # Set Franka as robot
        # # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # # Set actions for the specific robot type (franka)
        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     body_name="panda_hand",
        #     controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        #     scale=0.5,
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        # )

@configclass
class UR10CubeLiftEnvCfg_PLAY(UR10CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
