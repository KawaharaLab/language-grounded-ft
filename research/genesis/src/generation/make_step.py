import os
import imageio.v3 as iio
import numpy as np


CAPTURE_IMAGES = os.environ.get("GENESIS_CAPTURE_IMAGES", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

class EarlyDropDetected(RuntimeError):
    """Raised when the object is dropped before an intentional release."""


EARLY_DROP_MONITOR = False
INTENTIONAL_RELEASE = False
EARLY_DROP_STREAK = 0
EARLY_DROP_PENDING = False
CONTACT_LOSS_CONFIRM_STEPS = 5
TARGET_SUPPORT_LINK_IDXS = set()
OBSTACLE_LINK_IDXS = set()
TARGET_SUPPORT_ENTITIES = ()


def set_early_drop_monitor(enabled: bool):
    global EARLY_DROP_MONITOR, EARLY_DROP_STREAK, EARLY_DROP_PENDING
    EARLY_DROP_MONITOR = bool(enabled)
    EARLY_DROP_STREAK = 0
    EARLY_DROP_PENDING = False


def set_intentional_release(enabled: bool):
    global INTENTIONAL_RELEASE, EARLY_DROP_STREAK, EARLY_DROP_PENDING
    INTENTIONAL_RELEASE = bool(enabled)
    if INTENTIONAL_RELEASE:
        EARLY_DROP_STREAK = 0
        EARLY_DROP_PENDING = False


def set_object_contact_targets(target_tile=None, obstacle=None):
    """
    Register optional support/obstacle entities used for per-step contact logging.
    """
    global TARGET_SUPPORT_LINK_IDXS, OBSTACLE_LINK_IDXS, TARGET_SUPPORT_ENTITIES

    def _entities(value):
        if value is None:
            return ()
        if isinstance(value, (list, tuple, set)):
            return tuple(value)
        return (value,)

    def _link_indices(entities):
        return {int(link.idx) for entity in _entities(entities) for link in entity.links}

    # Contact records contain rigid-link indices, not entity indices.
    TARGET_SUPPORT_ENTITIES = _entities(target_tile)
    TARGET_SUPPORT_LINK_IDXS = _link_indices(target_tile)
    OBSTACLE_LINK_IDXS = _link_indices(obstacle)


def _target_support_angle_deg() -> float:
    """Return the first movable target support's hinge angle in degrees.

    Fixed supports have no DOFs and are represented as 0 degrees. Keeping the
    column numeric for every terrain avoids introducing NaNs into force windows.
    """
    for entity in TARGET_SUPPORT_ENTITIES:
        try:
            values = entity.get_dofs_position()
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=float).reshape(-1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if values.size:
            return float(np.rad2deg(values[0]))
    return 0.0


def get_bounding_box(gso_object):
    aabbs = gso_object.get_AABB().cpu().numpy()
    return aabbs[0].tolist() + aabbs[1].tolist()


def _as_int_set(values) -> set[int]:
    if values is None:
        return set()
    arr = values
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr).reshape(-1)
    out = set()
    for v in arr.tolist():
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out

def _execute_simulation_step(
    scene,
    cam,
    franka,
    df,
    deform_csv,
    photo_path,
    photo_interval,
    name,
    gso_object,
    gripper_force=0.0,
    force_photo=False,
    cam_wrist=None,
):
    """
    Executes a single step in the simulation, records data, and optionally saves images.
    This is an internal helper function consolidating logic from make_step and final_make_step.

    Args:
        force_photo (bool): If True, saves a photo regardless of the photo_interval.
    """

    scene.step()
    t = int(scene.t) - 1

    # Record robot state (DOFs and force/torque on end-effector links)
    dofs = franka.get_dofs_position().tolist()
    # Links 9 and 10 are the gripper fingers. 9 is the left finger, 10 is the right finger.
    links_f = franka.get_links_contact_force([9, 10], sensor=True)
    links_t = franka.get_links_contact_torque([9, 10], sensor=True)
    left_finger_force = links_f[0].tolist()
    right_finger_force = links_f[1].tolist()
    left_finger_torque = links_t[0].tolist()
    right_finger_torque = links_t[1].tolist()
    force_torques = left_finger_force + left_finger_torque + right_finger_force +right_finger_torque

    eef_pos = franka.get_links_pos([8, 9, 10]).flatten().tolist()
    finger_ctrl = franka.get_dofs_control_force([7, 8])
    finger_control = finger_ctrl.tolist()
    obj_com = gso_object.get_root_COM().tolist()
    obj_mass = [gso_object.get_mass()]
    obj_bounding_box = get_bounding_box(gso_object)

    obj_contacts = [0, 0, 0, 0, 0]
    obj_contact_info = gso_object.get_contacts()
    link_a = obj_contact_info.get("link_a", [])
    link_b = obj_contact_info.get("link_b", [])
    obj_contact_pairs = _as_int_set(link_a) | _as_int_set(link_b)
    if franka.get_link("left_finger").idx in obj_contact_pairs:
        obj_contacts[0] = 1
    if franka.get_link("right_finger").idx in obj_contact_pairs:
        obj_contacts[1] = 1
    if 0 in obj_contact_pairs:
        obj_contacts[2] = 1
    if TARGET_SUPPORT_LINK_IDXS.intersection(obj_contact_pairs):
        obj_contacts[3] = 1
    if OBSTACLE_LINK_IDXS.intersection(obj_contact_pairs):
        obj_contacts[4] = 1
    target_surface_angle_deg = _target_support_angle_deg()
    df.loc[len(df)] = (
        [scene.t]
        + force_torques
        + dofs
        + eef_pos
        + finger_control
        + obj_com
        + obj_mass
        + obj_bounding_box
        + obj_contacts
        + [target_surface_angle_deg]
    )

    global EARLY_DROP_STREAK, EARLY_DROP_PENDING
    if EARLY_DROP_MONITOR and not INTENTIONAL_RELEASE:
        fingers_in_contact = bool(obj_contacts[0] or obj_contacts[1])
        if fingers_in_contact:
            EARLY_DROP_STREAK = 0
            EARLY_DROP_PENDING = False
        else:
            EARLY_DROP_STREAK += 1
            if EARLY_DROP_STREAK >= CONTACT_LOSS_CONFIRM_STEPS:
                EARLY_DROP_PENDING = True

    # Save photos from main camera and optional wrist camera.
    if CAPTURE_IMAGES and (force_photo or (t % photo_interval == 0)):
        cam.set_pose(pos=(3.0, 0.0, 0.35), lookat=(0.0, 0.0, 0.35))
        rgb, _, _, _ = cam.render(rgb=True)
        if photo_path:
            filepath = os.path.join(photo_path, f"camera_0/{name}_{t:05d}.png")
            iio.imwrite(filepath, rgb)
        if cam_wrist is not None and photo_path:
            rgb_wrist, _, _, _ = cam_wrist.render(rgb=True)
            filepath_wrist = os.path.join(photo_path, f"camera_wrist/{name}_{t:05d}.png")
            iio.imwrite(filepath_wrist, rgb_wrist)
    if t % 100 == 0 or force_photo:
        print(f"Step: {t:05d} | Object: {name}")

    # # Return False to stop the simulation if forces are too high (indicating instability)
    # if abs(df.iloc[-1, 8]) > 100:
    #     return False
    if EARLY_DROP_MONITOR and not INTENTIONAL_RELEASE and EARLY_DROP_PENDING:
        raise EarlyDropDetected(
            f"Object lost finger contact for >={CONTACT_LOSS_CONFIRM_STEPS} consecutive steps; "
            f"terminated at step {int(scene.t)}."
        )

    return True

# Define the public-facing functions that call the internal helper
def make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=False, **kwargs)

def final_make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=True, **kwargs)
