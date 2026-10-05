import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_DIR = ROOT / "src" / "processing" / "rigid"
sys.path.insert(0, str(PROCESSING_DIR))
SPEC = importlib.util.spec_from_file_location(
    "rigid_annotations", PROCESSING_DIR / "1_create_annotations.py"
)
annotations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotations)


def test_first_sustained_state_rejects_short_contact_flicker():
    contact = np.array([True, True, False, False, False, False, True, False, False, False, False, False])

    assert annotations._first_sustained_state(contact, False, 0, confirm_steps=5) == 7


def test_rollout_bug_detects_com_teleport():
    force_df = pd.DataFrame(
        {
            "obj_COM_x": [0.4, 0.4, 0.51],
            "obj_COM_y": [0.4, 0.4, 0.4],
            "obj_COM_z": [0.1, 0.1, 0.1],
        }
    )

    assert annotations.detect_rollout_bug(force_df) == "COM step displacement > 0.05 m"


def test_rollout_bug_rejects_whole_episode_for_extreme_force():
    force_df = pd.DataFrame(
        {
            "obj_COM_x": [0.4, 0.4],
            "obj_COM_y": [0.4, 0.4],
            "obj_COM_z": [0.1, 0.1],
            "left_fx": [1.0, 100.0],
        }
    )

    assert annotations.detect_rollout_bug(force_df) == "contact-force component >= 100 N"


def test_gentle_wait_after_place_is_unannotated():
    force_segment = pd.DataFrame(
        {
            "obj_left_finger": np.ones(20, dtype=bool),
            "obj_right_finger": np.ones(20, dtype=bool),
        }
    )

    action = annotations._determine_action(
        start=100,
        force_segment=force_segment,
        first_support_contact_idx=80,
        first_grasp_contact_idx=None,
        first_lift_idx=None,
        accidental_release_idx=None,
        release_event_idx=None,
        release_midair=False,
        first_obstacle_contact_idx=None,
        lift_command_step=None,
        early_drop_case=False,
        place_mode="gentle",
        pressing=np.zeros(200, dtype=bool),
    )

    assert action is None


def _action(**overrides):
    values = {
        "start": 100,
        "force_segment": pd.DataFrame(
            {
                "obj_left_finger": np.ones(20, dtype=bool),
                "obj_right_finger": np.ones(20, dtype=bool),
            }
        ),
        "first_support_contact_idx": None,
        "first_grasp_contact_idx": None,
        "first_lift_idx": None,
        "accidental_release_idx": None,
        "release_event_idx": None,
        "release_midair": False,
        "first_obstacle_contact_idx": None,
        "lift_command_step": None,
        "early_drop_case": False,
        "place_mode": "gentle",
        "pressing": np.zeros(200, dtype=bool),
    }
    values.update(overrides)
    return annotations._determine_action(**values)


def test_place_event_takes_priority_over_pressing_in_same_window():
    pressing = np.zeros(200, dtype=bool)
    pressing[115:120] = True

    assert _action(first_support_contact_idx=110, pressing=pressing) == "place"


def test_place_then_release_is_distinct_from_press_then_release():
    pressing = np.zeros(200, dtype=bool)
    pressing[105:110] = True

    assert (
        _action(first_support_contact_idx=105, release_event_idx=115, pressing=pressing)
        == "place then release"
    )
    assert _action(release_event_idx=115, pressing=pressing) == "press then release"


def test_midair_release_has_its_own_action():
    assert _action(release_event_idx=115, release_midair=True) == "release midair"


def test_release_location_uses_confirmed_support_at_hand_opening():
    supported = np.zeros(20, dtype=bool)
    supported[8:] = True

    assert annotations._release_is_midair(supported, 7)
    assert not annotations._release_is_midair(supported, 8)


def test_weight_is_hidden_while_support_surface_carries_object():
    force_segment = pd.DataFrame(
        {
            "obj_left_finger": np.ones(20, dtype=bool),
            "obj_right_finger": np.ones(20, dtype=bool),
            "obj_mass": np.full(20, 0.3),
        }
    )
    supported = np.ones(20, dtype=bool)
    airborne = np.zeros(20, dtype=bool)

    assert not annotations._weight_is_observable(force_segment, supported)
    assert annotations._weight_is_observable(force_segment, airborne)
    assert annotations._build_annotation(
        "press", "stable", force_segment, weight_observable=False
    ) == "Pressing an object against a supporting surface with secure contact."


def test_place_and_press_describe_actual_support_surface_instead_of_outcome():
    size = 30
    force_df = pd.DataFrame(
        {
            "obj_target_tile": np.r_[np.zeros(5), np.ones(25)].astype(bool),
            "obj_plane": np.zeros(size, dtype=bool),
        }
    )
    pressing = np.zeros(size, dtype=bool)
    pressing[10:20] = True

    place_surface = annotations._support_surface_for_action(
        action="place",
        interaction="stable",
        start=0,
        force_df=force_df,
        metadata={"contact_logging_version": 2},
        place_mode="inclined_10",
        first_support_contact_idx=5,
        pressing=pressing,
    )
    press_surface = annotations._support_surface_for_action(
        action="press",
        interaction="stable",
        start=0,
        force_df=force_df,
        metadata={"contact_logging_version": 2},
        place_mode="step_2cm",
        first_support_contact_idx=5,
        pressing=pressing,
    )

    assert place_surface == "a gentle slope"
    assert press_surface == "a large step"
    assert annotations._target_surface_name("wobble") == "a wobbling surface"
    assert annotations._target_surface_name("gentle_simple") == "a fixed flat surface"
    assert annotations._support_condition("a fixed flat surface") == "fixed support"
    assert annotations._support_condition("a wobbling surface") == "wobbling support"
    assert annotations._support_condition("a gentle slope") is None
    text = annotations._build_annotation(
        "place", "stable", pd.DataFrame(), support_surface=place_surface
    )
    assert text == "Placing an object on a gentle slope with secure contact."
    assert "stable" not in text.lower()
    assert "topple" not in text.lower()

    assert annotations._build_annotation(
        "press", "stable", pd.DataFrame(), support_surface="a wobbling surface"
    ) == "Pressing an object against a wobbling surface with secure contact."
    assert annotations._build_annotation(
        "place then release",
        "stable",
        pd.DataFrame(),
        support_surface="a fixed flat surface",
    ) == (
        "Placing an object on a fixed flat surface with secure contact and then releasing it."
    )


def test_wobble_surface_name_uses_window_motion_then_nearest_static_angle():
    assert annotations._wobble_surface_name(
        pd.DataFrame({"target_surface_angle_deg": np.zeros(80)})
    ) == "a flat surface"
    assert annotations._wobble_surface_name(
        pd.DataFrame({"target_surface_angle_deg": np.full(80, 7.0)})
    ) == "a gentle slope"
    assert annotations._wobble_surface_name(
        pd.DataFrame({"target_surface_angle_deg": np.full(80, 16.0)})
    ) == "a moderate slope"
    assert annotations._wobble_surface_name(
        pd.DataFrame({"target_surface_angle_deg": np.full(80, 27.0)})
    ) == "a steep slope"
    assert annotations._wobble_surface_name(
        pd.DataFrame({"target_surface_angle_deg": np.linspace(0.0, 0.1, 80)})
    ) == "a wobbling surface"


def test_accidental_drop_is_an_interaction_and_preserves_intended_action():
    assert (
        _action(
            accidental_release_idx=115,
            early_drop_case=True,
            accidental_drop_action="place",
        )
        == "place"
    )
    force_segment = pd.DataFrame({"obj_mass": [0.3]})
    assert annotations._build_annotation(
        "place", "accidental drop", force_segment
    ) == "Accidentally dropped a heavy object while placing it."


def _interaction_segment(
    object_motion: np.ndarray,
    *,
    common_motion: np.ndarray | None = None,
    bimanual: np.ndarray | None = None,
) -> pd.DataFrame:
    size = len(object_motion)
    common_motion = np.zeros_like(object_motion) if common_motion is None else common_motion
    bimanual = np.ones(size, dtype=bool) if bimanual is None else bimanual
    left = common_motion + np.array([0.0, -0.04, 0.0])
    right = common_motion + np.array([0.0, 0.04, 0.0])
    com = common_motion + object_motion
    return pd.DataFrame(
        {
            "step": np.arange(1, size + 1, dtype=float),
            "obj_COM_x": com[:, 0],
            "obj_COM_y": com[:, 1],
            "obj_COM_z": com[:, 2],
            "right_finger_x": right[:, 0],
            "right_finger_y": right[:, 1],
            "right_finger_z": right[:, 2],
            "left_finger_x": left[:, 0],
            "left_finger_y": left[:, 1],
            "left_finger_z": left[:, 2],
            "obj_left_finger": bimanual,
            "obj_right_finger": bimanual,
        }
    )


def test_tangential_slip_uses_five_mm_per_second_threshold():
    size = 80
    # 4.0 mm over 0.79 s is 5.063 mm/s and therefore slip.
    object_motion = np.column_stack(
        [np.linspace(0.0, 0.004, size), np.zeros(size), np.zeros(size)]
    )
    force_segment = _interaction_segment(object_motion)

    measurements = annotations._interaction_measurements(force_segment)
    assert measurements["window_duration_s"] == pytest.approx(0.79)
    assert measurements["tangential_displacement_m"] == pytest.approx(0.004)
    assert measurements["tangential_speed_mps"] == pytest.approx(0.004 / 0.79)
    assert annotations._estimate_interaction(force_segment) == "slip"


def test_motion_along_grasp_axis_is_not_slip():
    size = 80
    # The grasp axis is y; normal motion along y must be projected out.
    object_motion = np.column_stack(
        [np.zeros(size), np.linspace(0.0, 0.02, size), np.zeros(size)]
    )
    assert annotations._estimate_interaction(_interaction_segment(object_motion)) == "stable"


def test_common_gripper_and_object_motion_is_not_slip():
    size = 80
    common_motion = np.column_stack(
        [np.linspace(0.0, 0.10, size), np.zeros(size), np.zeros(size)]
    )
    object_motion = np.zeros((size, 3))
    assert annotations._estimate_interaction(
        _interaction_segment(object_motion, common_motion=common_motion)
    ) == "stable"


def test_precontact_approach_is_excluded_from_slip_motion():
    size = 80
    object_motion = np.column_stack(
        [np.r_[np.linspace(0.10, 0.0, 70), np.zeros(10)], np.zeros(size), np.zeros(size)]
    )
    contact = np.r_[np.zeros(70, dtype=bool), np.ones(10, dtype=bool)]
    force_segment = _interaction_segment(object_motion, bimanual=contact)

    measurements = annotations._interaction_measurements(force_segment)
    assert measurements["observable"]
    assert measurements["tangential_displacement_m"] == pytest.approx(0.0)
    assert annotations._estimate_interaction(force_segment) == "stable"


def test_interaction_requires_five_consecutive_bimanual_contact_steps():
    size = 80
    object_motion = np.column_stack(
        [np.linspace(0.0, 0.02, size), np.zeros(size), np.zeros(size)]
    )
    contact = np.zeros(size, dtype=bool)
    contact[[5, 7, 9, 11, 13]] = True
    force_segment = _interaction_segment(object_motion, bimanual=contact)

    measurements = annotations._interaction_measurements(force_segment)
    assert measurements["bimanual_contact_steps"] == 5
    assert measurements["longest_bimanual_contact_run"] == 1
    assert not measurements["observable"]
    assert annotations._estimate_interaction(force_segment) is None
    assert annotations._build_annotation("hold", None, force_segment) == "Holding an object."


def test_brief_contact_gap_does_not_hide_stick_slip_displacement():
    size = 80
    object_motion = np.column_stack(
        [np.linspace(0.0, 0.01, size), np.zeros(size), np.zeros(size)]
    )
    contact = np.ones(size, dtype=bool)
    contact[35:37] = False
    force_segment = _interaction_segment(object_motion, bimanual=contact)

    assert annotations._interaction_measurements(force_segment)["observable"]
    assert annotations._estimate_interaction(force_segment) == "slip"


def test_pressing_requires_supported_finger_contact_and_sustained_motion():
    size = 20
    force_df = pd.DataFrame(
        {
            "obj_plane": np.ones(size, dtype=bool),
            "obj_tile": np.zeros(size, dtype=bool),
            "obj_left_finger": np.ones(size, dtype=bool),
            "obj_right_finger": np.zeros(size, dtype=bool),
            "eef_z": np.r_[np.ones(5), 1.0 - np.arange(15) * 0.001],
            "left_fz": np.zeros(size),
            "right_fz": np.zeros(size),
        }
    )

    pressing = annotations._pressing_series(
        force_df, metadata={}, place_idx=4, release_command_step=None
    )
    assert pressing[6:20].all()

    force_df["obj_left_finger"] = False
    assert not annotations._pressing_series(
        force_df, metadata={}, place_idx=4, release_command_step=None
    ).any()


def test_passive_supported_wait_is_not_press_but_commanded_press_is():
    size = 20
    force_df = pd.DataFrame(
        {
            "obj_plane": np.ones(size, dtype=bool),
            "obj_tile": np.zeros(size, dtype=bool),
            "obj_left_finger": np.ones(size, dtype=bool),
            "obj_right_finger": np.ones(size, dtype=bool),
            "eef_z": np.ones(size),
            "left_fz": np.full(size, 10.0),
            "right_fz": np.full(size, 10.0),
        }
    )

    passive = annotations._pressing_series(
        force_df,
        metadata={},
        place_idx=4,
        release_command_step=None,
    )
    commanded = annotations._pressing_series(
        force_df,
        metadata={},
        place_idx=4,
        release_command_step=18,
        press_command_step=6,
    )

    assert not passive.any()
    assert commanded[6:18].all()


def test_supported_hold_is_removed_after_five_confirmed_steps():
    short_contact = annotations._confirmed_mask(
        np.r_[np.zeros(8, dtype=bool), np.ones(4, dtype=bool), np.zeros(8, dtype=bool)]
    )
    sustained_contact = annotations._confirmed_mask(
        np.r_[np.zeros(8, dtype=bool), np.ones(5, dtype=bool), np.zeros(7, dtype=bool)]
    )

    assert not annotations._hold_has_sustained_support("hold", short_contact, 0, 20)
    assert annotations._hold_has_sustained_support("hold", sustained_contact, 0, 20)
    assert not annotations._hold_has_sustained_support("press", sustained_contact, 0, 20)
    assert not annotations._hold_has_sustained_support(
        "hold", sustained_contact, 0, 20, includes_accidental_drop=True
    )


def test_windows_without_contact_from_both_fingers_are_removed():
    force_df = pd.DataFrame(
        {
            "obj_left_finger": [False, True, False, False],
            "obj_right_finger": [False, False, False, False],
        }
    )

    assert not annotations._has_bimanual_contact(force_df, 0, 4)
    force_df.loc[2, "obj_right_finger"] = True
    assert annotations._has_bimanual_contact(force_df, 0, 4)


def test_terminal_window_captures_event_between_regular_stride_starts():
    force_df = pd.DataFrame(
        {
            "left_fx": np.ones(107),
            "left_fy": np.zeros(107),
            "left_fz": np.zeros(107),
            "right_fx": np.ones(107),
            "right_fy": np.zeros(107),
            "right_fz": np.zeros(107),
            "left_tx": np.zeros(107),
            "left_ty": np.zeros(107),
            "left_tz": np.zeros(107),
            "right_tx": np.zeros(107),
            "right_ty": np.zeros(107),
            "right_tz": np.zeros(107),
            "obj_left_finger": np.ones(107, dtype=bool),
            "obj_right_finger": np.ones(107, dtype=bool),
        }
    )
    windows = [{"start": 0}]

    annotations._append_terminal_window(force_df, windows, event_idx=106)

    assert windows == [{"start": 0}, {"start": 27}]


def test_legacy_target_contact_uses_tile_overlap_and_height():
    force_df = pd.DataFrame(
        {
            "obj_min_x": [0.43, 0.70],
            "obj_max_x": [0.47, 0.74],
            "obj_min_y": [0.43, 0.43],
            "obj_max_y": [0.47, 0.47],
            "obj_min_z": [0.004, 0.004],
        }
    )
    metadata = {"target_xy": {"x": 0.45, "y": 0.45}}

    np.testing.assert_array_equal(
        annotations._legacy_target_contact_series(force_df, metadata),
        [True, False],
    )
