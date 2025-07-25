# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#-----------> PLACE INTO <ISAACLAB_PATH>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/ur5e_rg2

from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg_MOD import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.universal_robots_ur5e_rg2 import UR5e_CFG  # isort: skip


from transforms3d.euler import euler2quat
from math import pi


@configclass
class UR5eCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.table.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),
                                                                   rot=euler2quat(0, 0, -90 * pi/180),
                                                                  )

        # Set UR5e as robot
        self.scene.robot = UR5e_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot,")
        self.scene.robot.init_state=ArticulationCfg.InitialStateCfg(
                                        joint_pos={
                                            # Arm joints
                                            "shoulder_pan_joint": 0.0,
                                            "shoulder_lift_joint": 0.0,
                                            "elbow_joint": 0.0,
                                            "wrist_1_joint": 0.0,
                                            "wrist_2_joint": 0.0,
                                            "wrist_3_joint": 0.0,
                                            # Gripper joints
                                            "left_inner_finger_joint": 0.0,
                                            "finger_joint": 0.0,
                                            #"left_inner_knuckle_finger_joint": 0.0,
                                            "right_outer_knuckle_joint": 0.0,
                                            #"right_inner_knuckle_finger_joint": 0.0,
                                            "right_inner_finger_joint": 0.0,
                                    },)

        # Set actions for the specific robot type (UR5e)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "right_outer_knuckle_joint"],
            open_command_expr={
                "finger_joint": -15.0, 
                "right_outer_knuckle_joint": -15.0
            },
            close_command_expr={
                "finger_joint": 15.0, 
                "right_outer_knuckle_joint": 15.0
            },
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "gripper_center"

        # # Set Franka as robot
        # self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # # Set actions for the specific robot type (franka)
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        # )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )
        # # Set the body name for the end effector
        # self.commands.object_pose.body_name = "panda_hand"


        '''# Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.815], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )'''


        # Set 3 objects in the scene
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.2, 0.815], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="omniverse://localhost/Library/melissa_and_doug_blocks/cuboid_L_2and3by8_YELLOW.usd",
                scale=(0.1, 0.1, 0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        '''self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.3, 0.815], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="omniverse://localhost/Library/melissa_and_doug_blocks/plummer_block.usd",
                scale=(0.1, 0.1, 0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )'''
        '''self.scene.object3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.4, 0.815], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="omniverse://localhost/Library/melissa_and_doug_blocks/cylinder.usd",
                scale=(0.1, 0.1, 0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )'''



        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5e_rg2_no_articulation_export/UR5e_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5e_rg2_no_articulation_export/gripper_center", 

                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1],
                    ),
                ),
            ],
        )

        # # Listens to the required transforms
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # self.scene.ee_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        #     debug_vis=False,
        #     visualizer_cfg=marker_cfg,
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        #             name="end_effector",
        #             offset=OffsetCfg(
        #                 pos=[0.0, 0.0, 0.1034],
        #             ),
        #         ),
        #     ],
        # )




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
