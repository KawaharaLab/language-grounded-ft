#!/usr/bin/env python3
"""Materialize pinned upstreams and overlays without modifying either."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / 'workspace'
PINS = {
    'genesis': '8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44',
    'patchtst': '204c21efe0b39603ad6e2ca640ef5896646ab1a9',
    'robotiq': '17533fe6faa9688f40a7ccc1663f59054c6dc3e2',
}

def export(name, destination):
    source = ROOT / 'third_party' / name
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PINS[name]:
        raise RuntimeError(f'{name}: expected {PINS[name]}, found {actual}')
    destination.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['git', '-C', str(source), 'archive', 'HEAD'], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode='r|') as stream:
        stream.extractall(destination, filter='data')
    if proc.wait(): raise RuntimeError(f'git archive failed: {name}')

def overlay(source, destination, extra_ignore=()):
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git', *extra_ignore))

def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): path.unlink()
    elif path.exists(): raise RuntimeError(f'Will not replace existing non-link {path}')
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)

def materialize(with_robot=True):
    inventory = WS/'.materialized-overlays.json'
    old = json.loads(inventory.read_text()) if inventory.exists() else {}
    current = {}
    for name, title in [('genesis', 'Genesis'), ('patchtst', 'PatchTST')]:
        prefix = 'research/' + name + '/'
        if name not in old:
            tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', 'HEAD', prefix], text=True).splitlines()
            old[name] = [p[len(prefix):] for p in tracked]
        current[name] = [str(p.relative_to(ROOT/prefix)) for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for relative in set(old[name]) - set(current[name]):
            stale = WS/title/relative
            if stale.is_file() or stale.is_symlink(): stale.unlink()
    for name, title in [('genesis','Genesis'),('patchtst','PatchTST')]:
        destination = WS/title
        export(name, destination)
        overlay(ROOT/'research'/name, destination)
    if with_robot:
        robot = WS/'ur3_robotiq_ros2'
        overlay(ROOT/'robot/ros2_src', robot/'src')
        driver = robot/'external/robotiq_2f_gripper_ros2'
        export('robotiq', driver)
        serial = ROOT/'third_party/robotiq/serial-ros2'
        if not (serial/'CMakeLists.txt').exists():
            raise RuntimeError('Run git submodule update --init --recursive third_party/robotiq')
        overlay(serial, driver/'serial-ros2', ('log', 'install', 'build'))
        git_marker = driver/'serial-ros2/.git'
        if git_marker.exists(): git_marker.unlink()
        local_overlay = ROOT/'.local/robot-overlays/robotiq_2f_gripper_ros2'
        applied = []
        for filename, target in [
            ('robotiq_2f_gripper_node.hpp','robotiq_2f_gripper_hardware/include/robotiq_2f_gripper_hardware/robotiq_2f_gripper_node.hpp'),
            ('robotiq_2f_gripper_node.cpp','robotiq_2f_gripper_hardware/src/robotiq_2f_gripper_node.cpp')]:
            source = local_overlay/filename
            if source.exists():
                shutil.copy2(source, driver/target)
                applied.append(filename)
        if applied:
            print(f"Applied local-only Robotiq overlay from {local_overlay}: {', '.join(applied)}")
        else:
            print(f'No local Robotiq overlay found at {local_overlay}; using the pinned upstream unchanged')
    for title, names in {
        'Genesis': ['data','fig','video','paper_review_20260916'],
        'PatchTST': ['data','results','saved_models'],
        **({'ur3_robotiq_ros2': ['data','artifacts']} if with_robot else {}),
    }.items():
        for name in names:
            target = ROOT/'artifacts'/title/name
            target.mkdir(parents=True, exist_ok=True)
            path = WS/title/name
            # git-exported research media directories are replaced only if empty.
            if path.is_dir() and not path.is_symlink() and not any(path.iterdir()): path.rmdir()
            link(path, target)
    link(WS/'artifacts', ROOT/'artifacts')
    for title, env in [('Genesis','simulation'),('PatchTST','learning')]:
        link(WS/title/'.venv', ROOT/'.venvs'/env)
    inventory.write_text(json.dumps(current, indent=2)+'\n')
    print(f'Materialized: {WS}')

def doctor(with_robot=True):
    failures = []
    system_links = []
    mounts_file = ROOT/'artifacts/local-links.json'
    mounts = json.loads(mounts_file.read_text())['links'] if mounts_file.exists() else []
    allowed = [Path(e['source']).resolve() for e in mounts]
    linked_artifacts = []
    names = PINS if with_robot else {name: pin for name, pin in PINS.items() if name != 'robotiq'}
    for name, pin in names.items():
        try:
            actual = subprocess.check_output(['git','-C',str(ROOT/'third_party'/name),'rev-parse','HEAD'], text=True).strip()
            if actual != pin: failures.append(f'{name}: incorrect pin')
        except subprocess.CalledProcessError: failures.append(f'{name}: not initialized')
    for directory in (WS, ROOT/'artifacts'):
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [n for n in dirs if n not in ('__pycache__','.git','.retired_independent_copies')]
            for name in dirs+files:
                p=Path(base)/name
                if p.is_symlink():
                    resolved=p.resolve()
                    if not resolved.is_relative_to(ROOT):
                        if resolved.is_relative_to('/opt/ros/humble') and p.is_relative_to(WS/'ur3_robotiq_ros2/install') and resolved.exists():
                            system_links.append(str(p.relative_to(ROOT)))
                        elif any(resolved == source or resolved.is_relative_to(source) for source in allowed) and resolved.exists():
                            linked_artifacts.append(str(p.relative_to(ROOT)))
                        else: failures.append(f'undeclared or missing external link: {p} -> {resolved}')
                    elif not p.exists() and p.name != '.venv': failures.append(f'broken link: {p}')
    report={'root':str(ROOT),'failures':failures,'hardware_executed':False,'declared_ros_system_links':len(system_links),'declared_artifact_links':len(linked_artifacts),'artifact_mode':'external-symlinks' if mounts else 'unconfigured'}
    print(json.dumps(report,indent=2)); return bool(failures)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['materialize','doctor'])
    ap.add_argument(
        '--without-robot', action='store_true',
        help='materialize/check only Genesis and PatchTST; do not require the Robotiq checkout',
    )
    args=ap.parse_args()
    with_robot = not args.without_robot
    return materialize(with_robot) if args.command=='materialize' else doctor(with_robot)

if __name__=='__main__': sys.exit(main())
