import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generation_make_step", ROOT / "src" / "generation" / "make_step.py"
)
make_step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_step)


class _Link:
    def __init__(self, index):
        self.idx = index


class _Entity:
    def __init__(self, entity_index, link_indices, dof_positions=None):
        self.idx = entity_index
        self.links = [_Link(index) for index in link_indices]
        self._dof_positions = dof_positions

    def get_dofs_position(self):
        if self._dof_positions is None:
            return np.array([])
        return np.asarray(self._dof_positions)


def test_contact_targets_store_link_indices_not_entity_indices():
    support = _Entity(3, [17, 18])
    obstacle = _Entity(4, [23])

    make_step.set_object_contact_targets([support], obstacle)

    assert make_step.TARGET_SUPPORT_LINK_IDXS == {17, 18}
    assert make_step.OBSTACLE_LINK_IDXS == {23}


def test_target_support_angle_uses_movable_support_and_fixed_defaults_to_zero():
    fixed_support = _Entity(3, [17])
    hinged_support = _Entity(4, [18], [np.deg2rad(-12.5)])

    make_step.set_object_contact_targets([fixed_support, hinged_support])
    assert make_step._target_support_angle_deg() == -12.5

    make_step.set_object_contact_targets(fixed_support)
    assert make_step._target_support_angle_deg() == 0.0
