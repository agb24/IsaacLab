"""Articulation Configuration for WidowX-250 robot. 

The Joint Names are: [waist, shoulder, elbow, wrist_angle, wrist_rotate, 
                      left_finger <<PRISMATIC>>, right_finger <<MIMIC>>]
The Link names are: [wx_250_base_link, wx_250_shoulder_link, wx_250_upper_arm_link,
                     wx_250_forearm_link, wx_250_wrist_link, wx_250_gripper_link,
                     wx_250_gripper_prop_link, 
                         ==> gripper_center_frame
                     wx_250_left_finger_link, wx_250_right_finger_link]
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Configuration
##

WIDOWX_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/home/abhara13/Documents/akshay_work/ur5e_ik_ws/src/wx250/wx250.usd",
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
        pos = (0.0, 0.0, 0.815),
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
        },
    ),
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=94.50081,
            damping=20.0, 
        ),
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=94.50081,
            damping=20.0, 
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=94.50081,
            damping=20.0, 
        ),
        "wrist_angle": ImplicitActuatorCfg(
            joint_names_expr=["wrist_angle"],
            effort_limit=50.0,
            velocity_limit=30.0,
            stiffness=94.50081,
            damping=20.0, 
        ),
        "wrist_rotate": ImplicitActuatorCfg(
            joint_names_expr=["wrist_rotate"],
            effort_limit=50.0,
            velocity_limit=30.0,
            stiffness=50.0,
            damping=7.5, 
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger"],
            effort_limit=50.0,
            velocity_limit=30.0,
            stiffness=50.0,
            damping=7.5,
        ),
    }
)


WIDOWX_TEST_PD_CFG = WIDOWX_CFG.copy()
WIDOWX_TEST_PD_CFG.spawn.rigid_props.disable_gravity = False
WIDOWX_TEST_PD_CFG.actuators["waist"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["waist"].damping *= 40
WIDOWX_TEST_PD_CFG.actuators["shoulder"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["shoulder"].damping *= 40
WIDOWX_TEST_PD_CFG.actuators["elbow"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["elbow"].damping *= 40
WIDOWX_TEST_PD_CFG.actuators["wrist_angle"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["wrist_angle"].damping *= 40
WIDOWX_TEST_PD_CFG.actuators["wrist_rotate"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["wrist_rotate"].damping *= 40
WIDOWX_TEST_PD_CFG.actuators["gripper"].stiffness *= 100
WIDOWX_TEST_PD_CFG.actuators["gripper"].damping *= 40



"""
# =====================================================================
# -----> Modifications for suitability to IsaacLab planning - adding new frame as EEF ref frame
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
import omni.usd

# --- CONFIG ---
articulation_root = "/wx250"  # Where your robot starts
parent_link_path = "/wx250/wx250_gripper_link/wx250_ee_arm_link/wx250_gripper_bar_link/wx250_fingers_link/wx250_ee_gripper_link"  # EEF link path
new_frame_name = "gripper_center_frame"  # Name of your custom frame
usd_save_path = "/home/abhara13/Documents/akshay_work/ur5e_ik_ws/src/wx250/wx250_with_eef_frame.usd"  # Path to save patched USD

# --- SETUP ---
stage = omni.usd.get_context().get_stage()
new_frame_path = f"{parent_link_path}/{new_frame_name}"

# Create the new frame under the parent
frame_prim = stage.DefinePrim(new_frame_path, "Xform")

# Add 90-degree Y rotation (optional, can be changed to any transform)
xform = UsdGeom.Xform(frame_prim)
xform.AddRotateYOp().Set(90.0)
xform.AddRotateZOp().Set(-90.0)

# Apply RigidBodyAPI and author attribute
rigid_api = UsdPhysics.RigidBodyAPI.Apply(frame_prim)
rigid_api.CreateRigidBodyEnabledAttr(True)

# Create an invisible collision cube (required for IsaacLab body registration)
collider_path = new_frame_path + "/collider"
cube = UsdGeom.Cube.Define(stage, collider_path)

xformable = UsdGeom.Xformable(cube)
existing_scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
scale_op = existing_scale_ops[0] if existing_scale_ops else xformable.AddScaleOp()
scale_op.Set((0.001, 0.001, 0.001))

UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())

# Create a FixedJoint connecting parent_link → new frame
joint_path = new_frame_path + "/fixed_joint"
joint_prim = stage.DefinePrim(joint_path, "PhysicsFixedJoint")
fixed_joint = UsdPhysics.FixedJoint(joint_prim)
fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_link_path)])
fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(new_frame_path)])

# --- Save to new USD file ---
stage.GetRootLayer().Export(usd_save_path)
print(f"✅ USD saved to: {usd_save_path}")

# =====================================================================
"""