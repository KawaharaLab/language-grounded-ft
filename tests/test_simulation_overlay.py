import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "research" / "genesis"
MANIFEST = ROOT / "manifests" / "simulation_sources.json"
PIN = "8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44"


def test_simulation_manifest_and_required_files():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["upstream"]["commit"] == PIN
    required = (
        "LICENSE",
        "MODIFICATIONS.md",
        "src/workspace_paths.py",
        "src/generation/run_genesis_sim.py",
        "src/generation/smoke_cpu.py",
        "src/generation/assets/inclined_ramp.obj",
        "src/processing/rigid/1_create_annotations.py",
        "src/processing/rigid/create_train_balanced_set.py",
        "src/processing/rigid/create_independent_eval_sets.py",
        "src/processing/rigid/create_support_contrast_sets.py",
        "src/experiments/audit_rq1_surface_supervised.py",
        "src/experiments/rq1_language_composition/prepare.py",
        "src/evaluation/generate_semantic_ft_examples.py",
        "src/evaluation/audit_real_object_contact_windows.py",
        "src/evaluation/evaluate_terrain_force_baseline_offline.py",
        "src/evaluation/evaluate_wobble_lift_and_lower.py",
        "src/evaluation/generate_pick_place_force_videos.py",
        "genesis/utils/array_class.py",
        "genesis/engine/solvers/rigid/constraint/solver.py",
        "genesis/engine/entities/rigid_entity/rigid_entity.py",
        "tests/rigid/test_contact_torque.py",
    )
    assert all((OVERLAY / name).is_file() for name in required)
    excluded = (
        "genesis/engine/couplers/sap_coupler.py",
        "tests/deformable/test_fem.py",
        "src/evaluation/benchmark_deformable_methods.py",
        "src/evaluation/benchmark_rigid_elastic.py",
        "src/evaluation/lift_and_lower_object_openai_vision.py",
        "src/experiments/rq1_action_interaction/experiment.py",
        "src/experiments/rq1_semantic_geometry/experiment.py",
    )
    assert not any((OVERLAY / name).exists() for name in excluded)


def test_overlay_has_no_machine_specific_absolute_paths_or_caches():
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", str(OVERLAY.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any("__pycache__" in Path(name).parts for name in tracked)
    assert not any(Path(name).suffix == ".pyc" for name in tracked)

    offenders = []
    for path in OVERLAY.rglob("*"):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.name == "LICENSE":
            continue
        text = path.read_text(errors="ignore")
        if "/home/user" in text or "/mnt/nvme" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders


def test_contact_torque_overlay_is_complete():
    array_class = (OVERLAY / "genesis/utils/array_class.py").read_text()
    solver = (OVERLAY / "genesis/engine/solvers/rigid/constraint/solver.py").read_text()
    entity = (OVERLAY / "genesis/engine/entities/rigid_entity/rigid_entity.py").read_text()
    assert "contact_torque: qd.Tensor" in array_class
    assert "contact_torque=V(" in array_class
    assert "contact_torque[contact_data_link_a" in solver
    assert "def get_links_contact_torque" in entity


def test_modified_upstream_files_have_change_notices():
    modified = (
        "genesis/engine/entities/rigid_entity/rigid_entity.py",
        "genesis/engine/solvers/rigid/constraint/solver.py",
        "genesis/utils/array_class.py",
    )
    for relative in modified:
        first_line = (OVERLAY / relative).read_text().splitlines()[0]
        assert "Modified from Genesis" in first_line
        assert "Apache-2.0" in first_line
