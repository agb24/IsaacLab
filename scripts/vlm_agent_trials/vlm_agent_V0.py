'''
CUSTOM AGENT :: AKSHAY
'''

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms

class CartesianCtrlUtils():
    def __init__(self,
                 scene, 
                 sim):
        self.robot = scene["robot"]
        self.robot_entity_cfg = SceneEntityCfg("robot", 
                                               joint_names=[".*"], 
                                               body_names=["wrist_3_link"])
        self.robot_entity_cfg.resolve(scene)
        # Initialize the IK Controller
        self._setup_controller(scene, sim) 

    def _setup_controller(self, scene, sim):
        # Set up the Inverse Kinematic solver
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose",
                                                  use_relative_mode=False,
                                                  ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, 
                                                           num_envs=scene.num_envs, 
                                                           device=sim.device)
    
    def get_joint_goals(self):
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
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                                    ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                              )
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, 
                                                        jacobian, joint_pos)
        return joint_pos_des
    

def run_simulator(env: gym.Env):
    #print(f"\n\n\n\n ---------------> {type(env)} \n\n\n\n")
    #print(f"Env Functions: {dir(env)}, Env Vars: {vars(env)}")
    sim = env.env.cfg.sim
    # Get the Scene from the Gym env
    scene: InteractiveScene = env.env.scene
    # Set up the Cartesian Controllers
    cartesian_utils = CartesianCtrlUtils(scene, sim)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    current_goal_idx = 0

    # Set up IK Commands
    robot = scene["robot"]
    ik_commands = torch.zeros(scene.num_envs, 
                              cartesian_utils.diff_ik_controller.action_dim, 
                              device=robot.device)
    ee_goals = torch.tensor([[0.5, 0.5, 1.0, 0.707, 0, 0.707, 0],
                             [0.5, -0.4, 1.0, 0.707, 0.707, 0.0, 0.0],
                             [0.5, 0, 1.2, 0.0, 1.0, 0.0, 0.0]
                            ], 
                            device=sim.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            '''actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            '''

            if count % 1000 == 0:
                # Reset Time
                count = 0
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                '''robot.reset()'''
                # Reset Actions
                ik_commands[:] = ee_goals[current_goal_idx]
                '''joint_pos_des = joint_pos[:, cartesian_utils.robot_entity_cfg.joint_ids].clone()'''
                # Reset Controller
                cartesian_utils.diff_ik_controller.reset()
                '''cartesian_utils.diff_ik_controller.set_command(ik_commands)'''
                
                env.reset()
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                print(f"[DEBUG]: RESETTING...... Current Goal IDX is {current_goal_idx}")
            else:
                joint_pos_des = cartesian_utils.get_joint_goals()
                actions = joint_pos_des
                # apply actions
                env.step(actions)
            

            # @@@@@@ Moved from the Reset time
            joint_pos_des = joint_pos[:, cartesian_utils.robot_entity_cfg.joint_ids].clone()
            cartesian_utils.diff_ik_controller.set_command(ik_commands)
            #

            count += 1

            # Obtain EEF World Pose from simulation
            ee_pose_w = robot.data.body_state_w[:, 
                                                cartesian_utils.robot_entity_cfg.body_ids[0], 
                                                0:7]
            # update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

    # close the simulator
    env.close()


def main():
    '''# Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])'''

    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Run the Simulator
    run_simulator(env)
    
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()