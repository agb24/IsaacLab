import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


class CustomIKUtils():
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