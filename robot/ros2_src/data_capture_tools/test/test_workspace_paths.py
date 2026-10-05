"""Offline tests for relocatable robot configuration paths."""

from pathlib import Path

from data_capture_tools.workspace_paths import resolve_workspace_path


def test_relative_path_uses_explicit_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("FT_WORKSPACE_ROOT", str(tmp_path))
    assert resolve_workspace_path("artifacts/robot/model.pth") == (
        tmp_path / "artifacts/robot/model.pth"
    ).resolve()


def test_absolute_path_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("FT_WORKSPACE_ROOT", str(tmp_path / "other"))
    absolute = tmp_path / "model.pth"
    assert resolve_workspace_path(absolute) == absolute.resolve()
