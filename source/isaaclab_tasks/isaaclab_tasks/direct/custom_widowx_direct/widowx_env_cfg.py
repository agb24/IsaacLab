"""
Custom Environment Config File for WidowX-250 ::: AKSHAY
"""

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg,    \
                            RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController,    \
                                 DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# CUSTOM IMPORTS
from isaaclab_assets.robots.widowx250 import WIDOWX_TEST_PD_CFG as WIDOWX_CFG


@configclass 
class WidowX250CustomEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    # -------- ACTION, OBSERVATION, STATE SPACES
    action_space = 6
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

    # --------- Robot Setup
    # Set WidowX-250 as robot
    robot = WIDOWX_CFG.replace(prim_path = "/World/envs/env_.*/Robot")
    robot.init_state = ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0),    #(-0.230, -0.118, 0.815),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos={
                            # Arm joints
                            "waist": 0.0,
                            "shoulder": 0.0, 
                            "elbow": 0.0, 
                            "wrist_angle": 0.0, 
                            "wrist_rotate": 0.0,
                            # Gripper joints 
                            "left_finger": 0.0,
                        },)

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Camera setup
    height, width = 480, 640
    camera_matrix = [[612.478, 0.0, 309.723], 
                     [0.0, 612.362, 245.359], 
                     [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    focus_distance = 1.2
    pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
    # Get the camera params
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm
    # Create the Camera Config
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb",
                    "distance_to_image_plane",
                    "instance_id_segmentation_fast",],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, 
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), 
                                       rot=(0.5, -0.5, 0.5, -0.5), 
                                       convention="ros"),
    )


    # TODO ::: Modify this to correct values
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0