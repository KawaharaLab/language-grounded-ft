import torch

import genesis as gs
import genesis.utils.geom as gu

from ..utils import assert_allclose


def test_link_contact_wrench_matches_per_contact_reconstruction(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    floor = scene.add_entity(
        gs.morphs.Box(
            size=(2.0, 2.0, 0.1),
            pos=(0.0, 0.0, -0.05),
            euler=(0.0, 0.0, 30.0),
            fixed=True,
        )
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.3, 0.0, 0.3),
        )
    )
    scene.build()

    for _ in range(100):
        scene.step()

    contacts = floor.get_contacts(with_entity=box)
    assert len(contacts["position"]) > 0

    link_idx = floor.link_start
    link_pos = floor.get_links_pos(relative=False)[0]
    expected_force = torch.zeros(3, dtype=contacts["position"].dtype, device=contacts["position"].device)
    expected_torque = torch.zeros_like(expected_force)
    for position, link_a, link_b, force_a, force_b in zip(
        contacts["position"],
        contacts["link_a"],
        contacts["link_b"],
        contacts["force_a"],
        contacts["force_b"],
    ):
        if int(link_a) == link_idx:
            expected_force += force_a
            expected_torque += torch.linalg.cross(position - link_pos, force_a)
        if int(link_b) == link_idx:
            expected_force += force_b
            expected_torque += torch.linalg.cross(position - link_pos, force_b)

    assert torch.linalg.vector_norm(expected_torque) > 1.0e-3
    assert_allclose(floor.get_links_contact_force(sensor=False)[0], expected_force, atol=1.0e-6)
    assert_allclose(floor.get_links_contact_torque(sensor=False)[0], expected_torque, atol=1.0e-6)

    link_quat = floor.get_links_quat(relative=False)[0]
    assert_allclose(
        floor.get_links_contact_force(sensor=True)[0],
        gu.inv_transform_by_quat(expected_force, link_quat),
        atol=1.0e-6,
    )
    assert_allclose(
        floor.get_links_contact_torque(sensor=True)[0],
        gu.inv_transform_by_quat(expected_torque, link_quat),
        atol=1.0e-6,
    )

    box.set_pos((0.3, 0.0, 1.0))
    scene.step()
    assert_allclose(floor.get_links_contact_force(sensor=False), 0.0, atol=1.0e-8)
    assert_allclose(floor.get_links_contact_torque(sensor=False), 0.0, atol=1.0e-8)
