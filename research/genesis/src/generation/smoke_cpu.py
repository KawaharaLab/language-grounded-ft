#!/usr/bin/env python3
"""Run a tiny headless CPU scene and verify the contact-wrench overlay."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("MUJOCO_GL", "0")

import genesis as gs
import torch


def main() -> None:
    gs.init(backend=gs.cpu, precision="64", seed=0, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
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
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.3, 0.0, 0.3))
    )
    scene.build()
    for _ in range(100):
        scene.step()

    contacts = floor.get_contacts(with_entity=box)
    if len(contacts["position"]) == 0:
        raise RuntimeError("CPU smoke scene produced no floor/box contact")

    force = floor.get_links_contact_force(sensor=False)[0]
    torque = floor.get_links_contact_torque(sensor=False)[0]
    if not torch.isfinite(force).all() or not torch.isfinite(torque).all():
        raise RuntimeError("CPU smoke scene produced a non-finite contact wrench")
    if float(torch.linalg.vector_norm(torque)) <= 1.0e-3:
        raise RuntimeError("Expected an off-center floor contact torque")

    print(
        json.dumps(
            {
                "backend": "cpu",
                "steps": 100,
                "contacts": len(contacts["position"]),
                "force_norm": float(torch.linalg.vector_norm(force)),
                "torque_norm": float(torch.linalg.vector_norm(torque)),
                "genesis_root": str(Path(gs.__file__).resolve().parents[1]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
