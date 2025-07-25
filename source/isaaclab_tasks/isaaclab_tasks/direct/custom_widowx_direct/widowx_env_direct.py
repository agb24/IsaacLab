"""
Custom Environment for WidowX-250 ::: AKSHAY
"""

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg 
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg,    \
                            RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


from transforms3d.euler import euler2quat
from math import pi
from typing import Tuple


# CUSTOM IMPORTS
from .widowx_env_cfg import WidowX250CustomEnvCfg


class WidowX250CustomEnv(DirectRLEnv):
    """
     pre-physics step calls
       |-- _pre_physics_step(action)
       |-- _apply_action()
     post-physics step calls
       |-- _get_dones()
       |-- _get_rewards()
       |-- _reset_idx(env_ids)
       |-- _get_observations()
    """
       
    cfg: WidowX250CustomEnvCfg

    def __init__(self, cfg: WidowX250CustomEnvCfg,
                 render_mode: str | None = None,
                 **kwargs):
        # This initializes the cfg, render_mode, SimulationContext, render interval,
        # and calls :::
        # self._setup_scene(), self.sim.reset(), self.scene.update()
        super().__init__(cfg, render_mode, **kwargs)

        # Get Delta Time dt 
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Auxiliary Variables
        # Soft Joint Pos limits docs ->
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData.soft_joint_pos_limits 
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, 
                                              self._robot.num_joints), device=self.device)
        
    def _setup_scene(self):
        """
        This function sets up the scene elements by calling elements from the 
        custom env config file.
        """

        # Ground Plane Setup
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
    
        # Robot Setup
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Add Camera
        camera = Camera(cfg=self.cfg.camera)
        self.scene.sensors["camera1"] = camera
                
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    # ------------------- pre-physics step calls -------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Function _pre_physics_step() is called within DirectRLEnv.step() BEFORE
        running every step. It is called before the _apply_action() function.

        Use this to pre-process actions before _apply_action is called.
        """
        
        self.robot_dof_targets[:] = torch.clamp(actions,
                                                self.robot_dof_lower_limits,
                                                self.robot_dof_upper_limits,
                                               )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    
    # ------------------- post-physics step calls -------------------
    def get_camera_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Camera data contains: 
        #   {"rgb": shape=1x480x640x3, 
        #    "distance_to_image_plane": shape=1x480x640x1, 
        #    "instance_segmentation_fast": shape=1x480x640x1}
        rgb_data = self.scene.sensors["camera1"].data.output["rgb"]
        inst_id_seg_data = self.scene.sensors["camera1"].data.output["instance_id_segmentation_fast"]
        return rgb_data, inst_id_seg_data

    def _get_observations(self):    # -> VecEnvObs
        """
        Function _get_observations() is called within DirectRLEnv.step() AFTER
        applying actions in each step, and saved into an Observations Buffer.
        """

        # Get Camera Data
        cam_rgb, cam_inst_id_seg = self.get_camera_data()
        # TODO Get the EEF Position

        # Create Observation Dict
        obs_dict = {"cam_rgb": cam_rgb,
                    "cam_inst_id_seg": cam_inst_id_seg,
                   }
        rl_obs_dict = {"policy": torch.cat([obs_dict[obs_key] for obs_key
                                                    in obs_dict.keys()],
                                                    # in ["cam_inst_seg"] ],
                                            dim = -1)
                      }
        
        return rl_obs_dict

    def _get_states(self):      # -> VecEnvObs
        return None
    
    def _get_rewards(self):
        """
        Function _get_rewards() is called within DirectRLEnv.step() AFTER
        applying actions in each step, and saved into a Rewards Buffer.
        """

        return 1

    def _get_dones(self):
        """
        Function _get_dones() is called within DirectRLEnv.step() AFTER
        applying actions in each step, and saved into a Reset Buffer.
        """

        terminated = self._robot.data.joint_pos[:, 3] > 0.5
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Function _reset_idx() is called within DirectRLEnv.step() AFTER 
        applying actions in each step.
        This function resets the robot positions of each 'Env' to default.
        """
        super()._reset_idx(env_ids)

        # Reset robot state to default
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, 
                                self.robot_dof_lower_limits, 
                                self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)