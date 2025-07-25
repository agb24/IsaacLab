from builtin_interfaces.msg import Duration

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY



class CartesianGlobalCtrl():
    def __init__(self):
        self.dh_params = [
            (0,  0.1625,  0,     np.pi/2),  
            (0,  0,      -0.425,  0),       
            (0,  0,      -0.3922, 0),       
            (0,  0.1333,  0,     np.pi/2),  
            (0,  0.0997,  0,    -np.pi/2),  
            (0,  0.0996,  0,     0)
        ]

    def rpy_to_matrix(self, rpy):
        return R.from_euler('xyz', rpy, degrees=True).as_matrix()
    
    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1]
        ])
    
    def forward_kinematics(self, dh_params, joint_angles, base_transform=None):
        if base_transform is None:
            T = np.eye(4)
        else:
            T = base_transform.copy()

        for i, (theta, d, a, alpha) in enumerate(dh_params):
            T_i = self.dh_transform(joint_angles[i] + theta, d, a, alpha)
            T = np.dot(T, T_i)
        return T
    
    def ik_objective(self, q, target_pose, base_transform=None):
        T_fk = self.forward_kinematics(self.dh_params, q, base_transform)
        pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
        rot_error = np.linalg.norm(T_fk[:3, :3] - target_pose[:3, :3])
        return 1.0 * pos_error + 0.1 * rot_error

    def compute_ik(self, position, rpy, 
                   q_guess=None, base_transform=None, 
                   max_tries=5, dx=0.001):
        if q_guess is None:
            q_guess = np.radians([85, -80, 90, -90, -90, 90])

        original_position = np.array(position)

        for i in range(max_tries):
            perturbed_position = original_position.copy()
            perturbed_position[0] += i * dx  # perturb x axis

            target_pose = np.eye(4)
            target_pose[:3, 3] = perturbed_position
            target_pose[:3, :3] = self.rpy_to_matrix(rpy)

            joint_bounds = [(-np.pi, np.pi)] * 6

            result = minimize(
                self.ik_objective,
                q_guess,
                args=(target_pose, base_transform),
                method='L-BFGS-B',
                bounds=joint_bounds
            )

            if result.success:
                return result.x

        print(f"IK failed after {max_tries} attempts. Tried perturbing from {original_position}.")
        return None




def make_point(joint_positions, seconds):
    return {
        "positions": [float(x) for x in joint_positions],  # ensures all are float
        "velocities": [0.0] * 6,
        "time_from_start": Duration(sec=int(seconds)),
    }

def home():
    joint_angles = compute_ik(HOME_POSE[0:3], HOME_POSE[3:6])
    if joint_angles is not None:
        return [make_point(joint_angles, 4)]
    return []

def move(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def moveZ(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def moveXY(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is not None:
        return [make_point(joint_angles, seconds)]
    return []

def pick_and_place(block_pose, slot_pose):
    """
    block_pose and slot_pose are each (position, rpy), where position = [x, y, z]
    and EE orienttaion is [r, p, y]
    """
    block_hover = block_pose[0].copy() ## copying positions
    block_hover[2] += 0.1  # hover 10cm above block

    slot_hover = slot_pose[0].copy()
    slot_hover[2] += 0.1  # hover 10cm above slot

    segment_duration = 6 # specify segment_duration

    return {
        "traj0": home(),
        "traj1": move(block_hover,block_pose[1],segment_duration), # hovers on block 
        "traj2": moveZ(block_pose[0],block_pose[1],segment_duration), # descends to grip position, 
        "traj3": moveZ(block_pose[0],block_pose[1],segment_duration), # gripper close
        "traj4": moveZ(block_hover,block_pose[1],segment_duration), # holds block and hovers 
        "traj5": moveXY(slot_hover,slot_pose[1],segment_duration), # holds block and moves in 2D to hover on slot
        "traj6": moveZ(slot_pose[0],slot_pose[1],segment_duration), # holds block and descends into slot,
        "traj7": moveZ(slot_pose[0],slot_pose[1],segment_duration), # gripper open
        "traj8": home() # homing
    }



if __name__ == "__main__":
    compute_ik = CartesianGlobalCtrl().compute_ik

    block_pose = ([0.09, -0.549, 0.18], [0, 180, 0])
    slot_pose = ([0.024, -0.375, 0.204], [0, 180, 0])
    trajectories = pick_and_place(block_pose, slot_pose)
    print(type(trajectories))

    [print(str(a)+"\n") for a in trajectories.items()]