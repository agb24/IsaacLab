'''
CUSTOM ENVIRONMENT CONFIG :: AKSHAY
'''

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg 
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
from isaaclab.utils.math import sample_uniform


from transforms3d.euler import euler2quat
from math import pi


# CUSTOM IMPORTS
#from isaaclab_assets.robots.universal_robots_ur5e_rg2 import UR5E_HIGH_PD_CFG as UR5e_CFG 
from isaaclab_assets.robots.universal_robots_ur5e_rg2 import UR5E_TEST_PD_CFG as UR5e_CFG


@configclass
class UR5eRG2CustomTableEnvCfg(DirectRLEnvCfg):
    # TODO ::: Modify this to correct values
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    # -------- ACTION, OBSERVATION, STATE SPACES
    action_space = 10
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
    # Set UR5e as robot
    robot = UR5e_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.88),    #(-0.230, -0.118, 0.815),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos={
                            # Arm joints
                            "shoulder_pan_joint": 0.0,
                            "shoulder_lift_joint": -1.0472,
                            "elbow_joint": 1.0472,
                            "wrist_1_joint": -1.5708,
                            "wrist_2_joint": -1.5708,
                            "wrist_3_joint": 0.0,
                            # Gripper joints
                            "finger_joint": 0.0,
                            "left_inner_finger_joint": 0.0,
                            #"left_inner_knuckle_finger_joint": 0.0,
                            "right_outer_knuckle_joint": 0.0,
                            #"right_inner_knuckle_finger_joint": 0.0,
                            "right_inner_finger_joint": 0.0,
                    },)

    # Table Setup
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), 
                                                rot=euler2quat(0, 0, 0)), #-90 * pi/180)),
        #spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        spawn=UsdFileCfg(usd_path="omniverse://localhost/Library/VentionAssembly_393251_v90_SCALE_ROT_STATIC.usd")
    )

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

    # Objects Setup
    # Set 3 objects in the scene
    object1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.2, 0.815), 
                                                  rot=(1.0, 0.0, 0.0, 0.0)),
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
    object2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.815), 
                                                  rot=(1.0, 0.0, 0.0, 0.0)),
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
    )
    object3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.4, 0.815), 
                                                  rot=(1.0, 0.0, 0.0, 0.0)),
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