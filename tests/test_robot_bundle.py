"""Source-only contracts for the real-robot bundle; no ROS graph is started."""

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "robot" / "ros2_src"


def test_expected_ros_packages_are_complete():
    expected = {
        "data_capture_tools", "mms101_driver", "gripper_client",
        "realsense_dual_launch", "teleop_servo",
    }
    assert expected == {path.parent.name for path in ROBOT.glob("*/package.xml")}


def test_research_license_declarations_and_texts():
    apache_packages = {
        "mms101_driver", "gripper_client", "realsense_dual_launch",
        "teleop_servo", "data_capture_tools",
    }
    for package in apache_packages:
        root = ET.parse(ROBOT / package / "package.xml").getroot()
        assert root.findtext("license") == "Apache-2.0"
    for package in apache_packages:
        text = (ROBOT / package / "LICENSE").read_text()
        assert "Apache License" in text and "Version 2.0" in text

    # The local authorization must not rewrite unresolved upstream metadata.
    upstream = ROOT / "third_party" / "robotiq"
    for package in ("robotiq_2f_gripper_hardware", "robotiq_2f_gripper_msgs"):
        root = ET.parse(upstream / package / "package.xml").getroot()
        assert root.findtext("license") == "TODO: License declaration"


def test_vendored_robotiq_visualization_keeps_upstream_bsd_notice():
    vendored = (
        ROBOT / "teleop_servo/config/robotiq_2f_140_gripper_visualization"
    )
    upstream = ROOT / "third_party/robotiq/robotiq_2f_gripper_description"
    assert (vendored / "LICENCE").read_bytes() == (
        upstream / "LICENCE"
    ).read_bytes()
    assert "ROS-Industrial" in (vendored / "LICENCE").read_text()
    assert "adapted" in (vendored / "NOTICE").read_text().lower()
    for mesh in (vendored / "meshes").rglob("*.stl"):
        relative = mesh.relative_to(vendored)
        assert mesh.read_bytes() == (upstream / relative).read_bytes(), relative


def test_robot_configs_are_yaml_and_have_no_developer_home_paths():
    for path in (ROBOT / "data_capture_tools" / "config").glob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text()), dict), path
        assert "/home/user" not in path.read_text(), path


def test_workspace_contract_and_manifests(monkeypatch, tmp_path):
    monkeypatch.setenv("FT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    package = ROBOT / "data_capture_tools"
    monkeypatch.syspath_prepend(str(package))
    from data_capture_tools.workspace_paths import resolve_workspace_path

    assert resolve_workspace_path("PatchTST/model.pth") == (
        tmp_path / "workspace" / "PatchTST" / "model.pth"
    ).resolve()
    for name in ("robot-dependencies.json", "robot-artifacts.json"):
        assert json.loads((ROOT / "manifests" / name).read_text())["schema_version"] == 1


def test_environment_recipe_is_source_only():
    recipe = (ROOT / "env" / "robot.sh").read_text()
    forbidden = ("ros2 launch", "ros2 run", "sudo ", "apt ", "colcon build")
    assert not any(command in recipe for command in forbidden)
    assert 'FT_WORKSPACE_ROOT="${FT_REPO_ROOT}/workspace"' in recipe
    assert "unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH" in recipe
    assert "unset PYTHONPATH LD_LIBRARY_PATH" in recipe
    assert ".venvs/learning/lib/python3.10/site-packages" in recipe


def test_no_build_trees_or_obvious_credentials_are_vendored():
    package_children = {
        child.name
        for package in ROBOT.iterdir()
        if package.is_dir()
        for child in package.iterdir()
    }
    assert not ({"build", "install", "log"} & package_children)
    sensitive_suffixes = (".pem", ".key", ".p12")
    assert not any(
        path.is_file() and path.name.lower().endswith(sensitive_suffixes)
        for path in (ROOT / "robot").rglob("*")
    )
