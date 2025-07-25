'''
CUSTOM AGENT :: AKSHAY

1. This will move to <<NUM>> positions, which are defined in the base frame of the robot.
        TODO: Move the Cartesian Controller to the Direct RL Environment.
              Send only the 7-DoF [XYZ RPY Gripper]  commands to the env.
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

import numpy as np



def run_simulator(env: gym.Env):
    #print(f"\n\n\n\n ---------------> {type(env)} \n\n\n\n")
    #print(f"Env Functions: {dir(env)}, Env Vars: {vars(env)}")
    sim_cfg = env.env.cfg.sim
    sim = env.env.sim

    # Get the Scene from the Gym env
    scene: InteractiveScene = env.env.scene
    # Set up the Cartesian Controllers
    cartesian_utils = CartesianCtrlUtils(scene, sim_cfg)

    # Visualization Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    '''world_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/World"))
    if VIEW_MARKERS == True:
        world_marker.visualize(torch.tensor([0, 0, 0]), torch.tensor([1, 0, 0, 0]))'''
    robot_base_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/base_link"))
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    current_goal_idx = 0
    # Set up IK Goal Commands
    robot = scene["robot"]
    ik_commands = torch.zeros(scene.num_envs, 
                              cartesian_utils.diff_ik_controller.action_dim, 
                              device=robot.device)
    ee_goals_w = torch.tensor([[0.35, 0.35, 1.2, 0.0, 1.0, 0.0, 0.0],
                             [-0.35, 0.35, 1.2, 0.0, 1.0, 0.0, 0.0],
                             [0.35, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0],
                             [-0.35, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0]
                            ],
                            device=sim_cfg.device)
    
    # Reset Controller
    cartesian_utils.diff_ik_controller.reset()
    cartesian_utils.diff_ik_controller.set_command(ik_commands)
    joint_pos_des = robot.data.joint_pos[:, 
                                         cartesian_utils.robot_entity_cfg.joint_ids].clone()

    # Print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Reset environment
    env.reset()
    # Get the physics time
    sim_dt = sim.get_physics_dt()

    # Set the Initial Goal
    ik_commands[:] = ee_goals_w[current_goal_idx].unsqueeze(0)
    cartesian_utils.set_goal_world(ik_commands[:, 0:3], ik_commands[:, 3:7])

    count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            '''actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            '''
            
            #  1.Get Jacobian  2.Get Curr Jt State  
            #  3.Get EEF pos in base frame  4.Compute Jt Cmds
            joint_pos_des, ee_pos_b, ee_quat_b = cartesian_utils.get_joint_goals()
            # 
            robot.set_joint_position_target(joint_pos_des, 
                                            joint_ids=cartesian_utils.robot_entity_cfg.joint_ids)
            # Get the goal positions
            goal_pos = ik_commands[:, 0:3]
            goal_quat = ik_commands[:, 3:7]
            # Get goal positions in Robot Co-ord
            root_pose_w = robot.data.root_state_w[:, 0:7]
            goal_pos_b, goal_quat_b = cartesian_utils.get_any_frame_in_base_frame(
                                                        root_pose_w, 
                                                        torch.cat((goal_pos, goal_quat), 
                                                                  dim=1),
                                                        )
            # Get Position Error
            pos_err = torch.norm(ee_pos_b - goal_pos_b)
            quat_err = 1.0 - torch.abs(torch.sum(ee_quat_b * goal_quat_b, dim=-1))


            if (count % 250 == 0) or (pos_err < 5e-3 and quat_err < 5e-2): #(pos_err < 0.1 and quat_err < 0.1):
                print(f"----------> Count: {count}, Pos Error: {pos_err}, Quat Error: {quat_err}")
                #print(f"Goal Pos: {goal_pos_b}, EE Pos: {ee_pos_b}")
                print(f"[Goal {current_goal_idx}] reached. Resetting...")

                # Reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                '''robot.write_joint_state_to_sim(joint_pos, joint_vel)'''
                # Change Goal ID
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals_w)
                # Run the Next Action
                ik_commands[:] = ee_goals_w[current_goal_idx].unsqueeze(0)
                # Reset Controller
                cartesian_utils.diff_ik_controller.reset()
                # change goal
                print(f"[DEBUG]: RESETTING...... Current Goal IDX is {current_goal_idx}")
            

            # ---------- SET THE GOAL IN WORLD CO-ORDINATES
            # @@NOTE@@ Moved from the Reset block
            '''joint_pos_des = joint_pos[:, cartesian_utils.robot_entity_cfg.joint_ids].clone()'''
            cartesian_utils.set_goal_world(ik_commands[:, 0:3], ik_commands[:, 3:7])
            # ---------- SET THE GOAL IN WORLD CO-ORDINATES


            scene.write_data_to_sim()
            # Perform Step
            sim.step()
            # Update Sim Time
            count += 1
            # Update Buffers
            scene.update(sim_dt)

            # Obtain EEF World Pose from simulation
            ee_pose_w = robot.data.body_state_w[:, 
                                                cartesian_utils.robot_entity_cfg.body_ids[0], 
                                                0:7]
            if VIEW_MARKERS == True:
                #robot_base_marker.visualize(ee_pos_b[:, 0:3], ee_quat_b[:, 3:7])
                # update marker positions
                ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

    # close the simulator
    env.close()


def main():
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