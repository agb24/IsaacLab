# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR10e with Robotiq 2F-140 gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UR10/ur10e_robotiq2f-140.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Arm joints
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Gripper joints
            "finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "ur10_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "left_inner_finger_joint",
                "left_inner_finger_pad_joint",
                "left_outer_finger_joint",
                "right_inner_finger_joint",
                "right_inner_finger_pad_joint",
                "right_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
            velocity_limit=1.0,
            effort_limit=50.0,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of UR10e with Robotiq 2F-140 gripper using implicit actuator models."""
