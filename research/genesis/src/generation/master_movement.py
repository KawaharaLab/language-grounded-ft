import numpy as np
from make_step import make_step


INTERPOLATE = 1
def set_to_pose(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 qpos, motors_dof, fingers_dof, steps=1, cam_wrist=None):
    """Moves the robot to a target joint configuration (qpos) over several steps."""
    for _ in range(steps):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name, cam_wrist=cam_wrist
        )

def descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                      end_effector, x, y, z, motors_dof, fingers_dof,
                      quat=np.array([0, 1, 0, 0]), steps=300, gripper_opening=0.04, cam_wrist=None):
    """Moves the end-effector down to a target position (x,y,z)."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    qpos[-2:] = gripper_opening
    path = franka.plan_path(qpos, num_waypoints=steps*INTERPOLATE)

    for waypoint in path:
        franka.control_dofs_position(waypoint)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name, cam_wrist=cam_wrist
        )

def grasp_object_position(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=50, cam_wrist=None):
    """
    Closes the gripper to grasp an object using position control.
    The gripper closes incrementally over a number of steps to ensure a secure grasp.
    0.04 is fully open, 0.0 is fully closed.
    """
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    gripper_pos = 0.04
    for i in range(steps*INTERPOLATE):
        gripper_pos -= 0.04 / (steps*INTERPOLATE)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([gripper_pos, gripper_pos]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force, cam_wrist=cam_wrist
        )

def lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                quat=np.array([0, 1, 0, 0]), steps=1, cam_wrist=None):
    """Lifts the object vertically from its current position."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    # franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
    franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
    return make_step(
        scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
        photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
        gripper_force=grip_force, cam_wrist=cam_wrist
    )

def move_to_place_xy(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, motors_dof, fingers_dof, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=300, cam_wrist=None):
    """Moves to target XY while keeping the current end-effector height fixed."""
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_fixed = eef_pos[2]
    x_start, y_start = eef_pos[0], eef_pos[1]
    total_steps = max(1, steps * INTERPOLATE)

    for step_idx in range(total_steps):
        alpha = (step_idx + 1) / total_steps
        x_target = (1.0 - alpha) * x_start + alpha * x
        y_target = (1.0 - alpha) * y_start + alpha * y
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x_target, y_target, z_fixed]),
            quat=quat,
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force, cam_wrist=cam_wrist
        )

def release_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 fingers_dof, grip_force, steps=50, cam_wrist=None):
    finger_pos = franka.get_dofs_position(fingers_dof).cpu().numpy()
    finger_margin = np.array([0.04, 0.04]) - finger_pos
    for _ in range(steps*INTERPOLATE):
        finger_pos += finger_margin / steps
        franka.control_dofs_position(finger_pos, fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force, cam_wrist=cam_wrist
        )
    for _ in range (70):
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force, cam_wrist=cam_wrist
        )

def descend_to_place_cautiously(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                        quat=np.array([0, 1, 0, 0]), steps=300, cam_wrist=None):
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_hover = eef_pos[2]
    for step in range(steps*INTERPOLATE):
        z_target = z_hover - (z_hover - z) * (step + 1) / (steps*INTERPOLATE)
        qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z_target]), quat=quat)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
        # franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force, cam_wrist=cam_wrist
        )
