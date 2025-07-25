'''
CARTESIAN CONTROLLER :: ISAAC DEFAULT (AKSHAY)
'''

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import torch


class CartesianCtrl():
    def __init__(self,
                 scene, 
                 sim_cfg):
        self.robot = scene["robot"]
        self.robot_entity_cfg = SceneEntityCfg("robot", 
                                               joint_names=[".*"], 
                                               body_names=["wrist_3_link"])
        self.robot_entity_cfg.resolve(scene)
        # Initialize the IK Controller
        self._setup_controller(scene, sim_cfg) 

    def _setup_controller(self, scene, sim_cfg):
        # Set up the Inverse Kinematic solver
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose",
                                                  use_relative_mode=False,
                                                  ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, 
                                                           num_envs=scene.num_envs, 
                                                           device=sim_cfg.device)
    
    def set_goal_world(self, goal_pos_w, goal_quat_w):
        """
        Set the overall goal for the Cartesian Controller.
        """
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
                                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                                        goal_pos_w, goal_quat_w
                                  )
        goal_pose_b = torch.cat([goal_pos_b, goal_quat_b], dim=-1)
        self.diff_ik_controller.set_command(goal_pose_b)

    def get_any_frame_in_base_frame(self, root_pose_w, any_pose_w):
        any_pos_b, any_quat_b = subtract_frame_transforms(
                                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                                    any_pose_w[:, 0:3], any_pose_w[:, 3:7]
                              )
        return any_pos_b, any_quat_b

    def get_joint_delta_subgoals(self):
        """
        1. Get Jacobian  
        2. Get Curr Jt State  
        3. Get EEF pos in base frame  
        4. Compute & Return Jt Delta Cmds towards reaching overall goal
        """

        robot = self.robot
        robot_entity_cfg = self.robot_entity_cfg
        # ----------- Get robot parameters
        # Obtain the frame index of the EEF -> For fixed-base, frame index is (body_index -1)
        if robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        # Compute values required for IK 
        # Jacobian shape: (articulation_index, jacobian_shape.numCols, jacobian_shape.numRows)
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, 
                                                                robot_entity_cfg.joint_ids]
        # EEF in World frame: [pos, quat, lin_vel, ang_vel] -> Shape (num_inst, num_bodies, 13)
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # Robot Root State in World frame: [pos, quat, lin_vel, ang_vel] -> Shape (num_inst, 13)
        root_pose_w = robot.data.root_state_w[:, 0:7]
        # Joint State of the robot -> Shape (num_instances, num_joints)
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        # Compute EEF in Base (Robot root) frame
        ee_pos_b, ee_quat_b = self.get_any_frame_in_base_frame(root_pose_w, ee_pose_w)
        # Compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, 
                                                        jacobian, joint_pos)
        return joint_pos_des, ee_pos_b, ee_quat_b