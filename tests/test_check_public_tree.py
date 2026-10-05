import importlib.util
from pathlib import Path
import subprocess
import sys

MODULE_PATH = Path(__file__).parents[1] / "tools" / "check_public_tree.py"
SPEC = importlib.util.spec_from_file_location("check_public_tree", MODULE_PATH)
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)

def run(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

def commit(repo, message):
    run(repo, "add", "-A")
    run(repo, "-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "-m", message)

def test_scans_deleted_history_without_printing_secret(tmp_path):
    run(tmp_path, "init", "-q")
    secret = "sk-" + "A" * 32
    credential = tmp_path / "settings.py"
    credential.write_text(f'api_key = "{secret}"\n')
    commit(tmp_path, "add credential")
    old_revision = run(tmp_path, "rev-parse", "HEAD").stdout.strip()
    credential.write_text('api_key = "${OPENAI_API_KEY}"\n')
    commit(tmp_path, "remove credential")
    findings, _, _ = scanner.scan_repository(tmp_path)
    assert any(f.revision == old_revision and f.path == "settings.py" for f in findings)
    rendered = "\n".join(f"{f.revision}:{f.path}: {f.reason}" for f in findings)
    assert secret not in rendered

def test_flags_private_artifacts_but_allows_small_synthetic_assets(tmp_path):
    run(tmp_path, "init", "-q")
    (tmp_path / "research" / "assets").mkdir(parents=True)
    (tmp_path / "research" / "assets" / "ramp.obj").write_text("v 0 0 0\n")
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "synthetic.csv").write_text("force,label\n0,none\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "trial.ckpt").write_bytes(b"tiny")
    (tmp_path / "measurements.csv").write_text("private,trial\n1,2\n")
    commit(tmp_path, "fixtures")
    findings, _, _ = scanner.scan_repository(tmp_path)
    assert not any(f.path.endswith("ramp.obj") for f in findings)
    assert not any(f.path.endswith("synthetic.csv") for f in findings)
    assert any(f.path == "models/trial.ckpt" for f in findings)
    assert any(f.path == "measurements.csv" for f in findings)

def test_scans_unstaged_tracked_worktree_content(tmp_path):
    run(tmp_path, "init", "-q")
    config = tmp_path / "config.yaml"
    config.write_text("token: ${SERVICE_TOKEN}\n")
    commit(tmp_path, "safe config")
    config.write_text('access_token: "abcdefghijklmnop-secret"\n')
    findings, _, _ = scanner.scan_repository(tmp_path)
    assert any(f.revision == "WORKTREE" and f.path == "config.yaml" for f in findings)

def test_detects_argparse_credential_default(tmp_path):
    run(tmp_path, "init", "-q")
    source = tmp_path / "cli.py"
    source.write_text('parser.add_argument("--api-key", default="abcdefghijklmnop-secret")\n')
    commit(tmp_path, "unsafe default")
    findings, _, _ = scanner.scan_repository(tmp_path)
    assert any(f.reason == "literal credential in call/default" for f in findings)

def test_rejects_external_tracked_symlink_without_following_it(tmp_path):
    run(tmp_path, "init", "-q")
    outside = tmp_path.parent / "outside-private-file"
    outside.write_text("contents must not be scanned")
    (tmp_path / "external").symlink_to(outside)
    commit(tmp_path, "external link")
    findings, _, _ = scanner.scan_repository(tmp_path, ["HEAD"])
    assert any(f.revision == "WORKTREE" and f.reason == "external symlink" for f in findings)

def test_ref_limits_history_to_parentless_public_snapshot(tmp_path):
    run(tmp_path, "init", "-q")
    secret = "ghp_" + "A" * 36
    (tmp_path / "private.py").write_text(f'auth_token = "{secret}"\n')
    commit(tmp_path, "private local history")
    run(tmp_path, "branch", "preserved-local")
    run(tmp_path, "switch", "--orphan", "public")
    (tmp_path / "private.py").unlink(missing_ok=True)
    (tmp_path / "public.py").write_text('auth_token = "${SERVICE_TOKEN}"\n')
    commit(tmp_path, "public snapshot")

    public_findings, _, _ = scanner.scan_repository(tmp_path, ["HEAD"])
    full_findings, _, _ = scanner.scan_repository(tmp_path)

    assert not any(f.path == "private.py" for f in public_findings)
    assert any(f.path == "private.py" for f in full_findings)

def test_forbids_unlicensed_overlay_source_path():
    path = "robot/overlays/robotiq_2f_gripper_ros2/robotiq_2f_gripper_node.cpp"
    assert "unlicensed source copy" in scanner.path_reasons(path, 10)

def test_rejects_private_local_cache_and_env_variants():
    assert scanner.path_reasons(".local/source.cpp", 10)
    assert scanner.path_reasons("project/.cache/result.bin", 10)
    assert "credential file" in scanner.path_reasons(".env.production", 10)
