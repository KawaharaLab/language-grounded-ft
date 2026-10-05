#!/usr/bin/env python3
"""Audit the tracked worktree and every blob reachable from Git refs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import argparse
import os
import ast
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_BLOB = 10 * 1024 * 1024
PRIVATE_COMPONENTS = {"artifacts", "checkpoints", "data", "datasets", "logs", "models", "outputs", "runs", "wandb", "workspace", ".cache", ".local", ".runtime", ".venvs"}
PRIVATE_SUFFIXES = {".bag", ".ckpt", ".csv", ".db3", ".feather", ".h5", ".hdf5", ".joblib", ".log", ".mcap", ".npy", ".npz", ".onnx", ".parquet", ".pickle", ".pkl", ".pt", ".pth", ".safetensors", ".tfrecord", ".tsv"}
PRIVATE_FILENAMES = {".env", ".env.local", ".netrc", "credentials.json"}
FORBIDDEN_PATHS = {
    "robot/overlays/robotiq_2f_gripper_ros2/robotiq_2f_gripper_node.cpp",
    "robot/overlays/robotiq_2f_gripper_ros2/robotiq_2f_gripper_node.hpp",
}
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("OpenAI API key", re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Slack token", re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}")),
)
SENSITIVE_NAMES = "api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|secret[_-]?key"
LITERAL_ASSIGNMENT = re.compile(rf"(?im)(?:^|[{{,;])\s*['\"]?(?:{SENSITIVE_NAMES})['\"]?\s*[:=]\s*['\"]([^'\"\r\n]{{16,}})['\"]")
PLACEHOLDER = re.compile(r"(?i)^(?:<[^>]+>|\$\{?\w+\}?|your[_ -]|example|dummy|fake|placeholder|replace[_ -]|x{8,})")

@dataclass(frozen=True, order=True)
class Finding:
    revision: str
    path: str
    reason: str

def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)

def path_reasons(path: str, size: int) -> list[str]:
    posix = PurePosixPath(path)
    lowered = tuple(part.lower() for part in posix.parts)
    reasons = []
    if path in FORBIDDEN_PATHS:
        reasons.append("unlicensed source copy")
    if any(part in PRIVATE_COMPONENTS for part in lowered):
        reasons.append("private/generated artifact path")
    if posix.name.lower() in PRIVATE_FILENAMES or posix.name.lower().startswith(".env."):
        reasons.append("credential file")
    synthetic_fixture = (
        size <= 64 * 1024
        and any(part in {"test", "tests"} for part in lowered)
        and any(part in {"fixture", "fixtures", "synthetic"} for part in lowered)
    )
    if posix.suffix.lower() in PRIVATE_SUFFIXES and not synthetic_fixture:
        reasons.append("data/model/trial artifact")
    if size > MAX_PUBLIC_BLOB:
        reasons.append("unexpected large file")
    return reasons

def content_reasons(data: bytes, path: str) -> list[str]:
    reasons = [reason for reason, pattern in SECRET_PATTERNS if pattern.search(data)]
    if b"\0" in data[:8192]:
        return reasons
    text = data.decode("utf-8", errors="replace")
    for match in LITERAL_ASSIGNMENT.finditer(text):
        if not PLACEHOLDER.match(match.group(1).strip()):
            reasons.append("literal credential assignment")
            break
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            reasons.append("invalid Python syntax")
        else:
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                sensitive_flag = any(
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and re.search(SENSITIVE_NAMES, arg.value, re.IGNORECASE)
                    for arg in call.args
                )
                for keyword in call.keywords:
                    value = keyword.value
                    sensitive_name = bool(keyword.arg and re.fullmatch(SENSITIVE_NAMES, keyword.arg, re.IGNORECASE))
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        literal = value.value.strip()
                        if len(literal) >= 16 and not PLACEHOLDER.match(literal) and (sensitive_name or (sensitive_flag and keyword.arg == "default")):
                            reasons.append("literal credential in call/default")
                            break
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [target.id.lower() for target in targets if isinstance(target, ast.Name)]
                literal = value.value.strip()
                if any(re.fullmatch(SENSITIVE_NAMES, name) for name in names) and len(literal) >= 16 and not PLACEHOLDER.match(literal):
                    reasons.append("literal credential assignment")
                    break
    return list(dict.fromkeys(reasons))

def tree_entries(repo: Path, revision: str):
    raw = git(repo, "ls-tree", "-r", "-z", "--long", revision)
    for record in filter(None, raw.split(b"\0")):
        metadata, path = record.split(b"\t", 1)
        _mode, kind, oid, size = metadata.split()
        if kind == b"blob":
            yield oid.decode(), int(size), path.decode("utf-8", errors="surrogateescape")

def inspect_blob(data: bytes, path: str, size: int, revision: str) -> list[Finding]:
    reasons = path_reasons(path, size) + content_reasons(data, path)
    return [Finding(revision, path, reason) for reason in dict.fromkeys(reasons)]

def scan_repository(repo: Path = ROOT, refs: list[str] | None = None) -> tuple[list[Finding], int, int]:
    findings: set[Finding] = set()
    revision_args = refs if refs else ["--all"]
    commits = git(repo, "rev-list", *revision_args).decode().splitlines()
    tracked_paths = [p for p in git(repo, "ls-files", "-z").split(b"\0") if p]
    for encoded_path in tracked_paths:
        path = encoded_path.decode("utf-8", errors="surrogateescape")
        absolute = repo / path
        if absolute.is_symlink():
            target = os.readlink(absolute)
            resolved = (absolute.parent / target).resolve()
            try:
                resolved.relative_to(repo.resolve())
            except ValueError:
                findings.add(Finding("WORKTREE", path, "external symlink"))
            data = target.encode("utf-8", errors="surrogateescape")
            findings.update(inspect_blob(data, path, len(data), "WORKTREE"))
        elif absolute.is_file():
            data = absolute.read_bytes()
            findings.update(inspect_blob(data, path, len(data), "WORKTREE"))
    blob_cache: dict[str, bytes] = {}
    for revision in commits:
        for oid, size, path in tree_entries(repo, revision):
            if oid not in blob_cache:
                blob_cache[oid] = git(repo, "cat-file", "blob", oid)
            findings.update(inspect_blob(blob_cache[oid], path, size, revision))
    return sorted(findings), len(tracked_paths), len(blob_cache)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", action="append", dest="refs", metavar="REVISION",
        help="audit commits reachable from this revision (repeatable; default: all refs)",
    )
    args = parser.parse_args()
    findings, tracked_count, blob_count = scan_repository(refs=args.refs)
    for finding in findings:
        print(f"{finding.revision}:{finding.path}: {finding.reason}")
    print(f"Checked {tracked_count} tracked paths and {blob_count} unique blobs across all reachable revisions; {len(findings)} issues")
    return bool(findings)

if __name__ == "__main__":
    sys.exit(main())
