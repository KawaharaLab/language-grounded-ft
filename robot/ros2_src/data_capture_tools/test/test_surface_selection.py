"""Tests for threshold-free two-surface selection."""

import math

import numpy as np
import pytest

from data_capture_tools.two_surface_selection_capture_node import (
    select_more_fixed_surface,
    slew_limit_vector,
    transform_pose_error_to_command_xy,
)


def test_selects_larger_fixed_minus_wobble_score():
    assert select_more_fixed_surface({"fixed": 0.12, "wobble": -0.03}) == "fixed"
    assert select_more_fixed_surface({"fixed": -0.20, "wobble": 0.05}) == "wobble"


@pytest.mark.parametrize(
    "scores",
    [
        {"fixed": 0.1},
        {"fixed": 0.1, "wobble": 0.1},
        {"fixed": math.nan, "wobble": 0.0},
    ],
)
def test_rejects_incomplete_tied_or_nonfinite_scores(scores):
    with pytest.raises(ValueError):
        select_more_fixed_surface(scores)


def test_transforms_ur_base_xy_error_to_base_link_command():
    command = transform_pose_error_to_command_xy(
        np.asarray([0.12, -0.08]),
        np.asarray([-1.0, -1.0]),
    )
    assert command.tolist() == pytest.approx([-0.12, 0.08])


def test_cartesian_command_slew_limit():
    command = slew_limit_vector(
        previous=np.asarray([0.0, 0.0, 0.0]),
        target=np.asarray([0.8, -0.8, 1.0]),
        rate_limit=np.asarray([1.5, 1.5, 2.0]),
        dt_sec=0.01,
    )
    assert command.tolist() == pytest.approx([0.015, -0.015, 0.02])
