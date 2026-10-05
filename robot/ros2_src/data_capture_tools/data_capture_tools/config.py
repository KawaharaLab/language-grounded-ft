"""Utilities for loading and validating data capture configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml

from .workspace_paths import resolve_workspace_path


def validate_object_name(value: str) -> str:
    """Validate a path-safe object label used by place evaluation runs."""
    object_name = str(value).strip()
    if not object_name:
        return ""
    if len(object_name) > 64:
        raise ValueError("object_name must be 64 characters or fewer")
    if object_name.startswith("place_"):
        raise ValueError(
            "object_name must not include the 'place_' prefix; "
            "pass only the object label"
        )
    if not object_name[0].isalnum() or not object_name[-1].isalnum():
        raise ValueError(
            "object_name must start and end with a letter or number"
        )
    if any(
        not (character.isalnum() or character == "_")
        for character in object_name
    ):
        raise ValueError(
            "object_name may contain only letters, numbers, and underscores"
        )
    return object_name


def validate_surface_condition(value: str) -> str:
    """Validate a path-safe support/surface label for place evaluations."""
    surface_condition = str(value).strip()
    if not surface_condition:
        return ""
    if len(surface_condition) > 32:
        raise ValueError("surface_condition must be 32 characters or fewer")
    if not surface_condition[0].isalnum() or not surface_condition[-1].isalnum():
        raise ValueError(
            "surface_condition must start and end with a letter or number"
        )
    if any(
        not (character.isalnum() or character == "_")
        for character in surface_condition
    ):
        raise ValueError(
            "surface_condition may contain only letters, numbers, and underscores"
        )
    return surface_condition


@dataclass
class TopicSpec:
    """Configuration for a topic that should be captured."""

    name: str
    type: str
    mode: str  # "image" or "csv"
    encoding: Optional[str] = None  # Only used for image topics


@dataclass
class BagRecorderConfig:
    """rosbag2 recording options."""

    storage: str = "sqlite3"
    compression_mode: Optional[str] = None
    compression_format: Optional[str] = None


@dataclass
class CaptureConfig:
    """Top-level configuration for a capture session."""

    output_root: Path
    task_name: str
    prompt: str
    viewer_topics: list[str]
    image_throttle_hz: float
    other_throttle_hz: float
    enable_image_compression: bool
    stop_key: str
    discard_key: str
    topics: List[TopicSpec]
    bag: BagRecorderConfig
    raw: dict[str, Any]
    source_path: Path


def _validate_topic(entry: dict[str, Any]) -> TopicSpec:
    if "name" not in entry or "type" not in entry or "mode" not in entry:
        raise ValueError(
            "Each topic entry must contain 'name', 'type', and 'mode' fields"
        )

    mode = entry["mode"].lower()
    if mode not in {"image", "csv"}:
        raise ValueError("Topic mode must be either 'image' or 'csv'")

    encoding = entry.get("encoding") if mode == "image" else None

    return TopicSpec(
        name=str(entry["name"]),
        type=str(entry["type"]),
        mode=mode,
        encoding=encoding,
    )


def load_capture_config(config_path: str | Path) -> CaptureConfig:
    """Load and validate a capture configuration YAML file."""

    path = resolve_workspace_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Capture config '{path}' does not exist")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    topics_data = data.get("topics")
    if not topics_data:
        raise ValueError("Config must list at least one topic under 'topics'")

    topics = [_validate_topic(entry) for entry in topics_data]

    bag_cfg = data.get("bag_record", {})
    bag = BagRecorderConfig(
        storage=bag_cfg.get("storage", "sqlite3"),
        compression_mode=bag_cfg.get("compression_mode"),
        compression_format=bag_cfg.get("compression_format"),
    )

    output_root = resolve_workspace_path(data.get("output_root", "data/robot"))
    task_name = str(data.get("task_name", "pick_and_place"))
    prompt = str(
        data.get(
            "prompt",
            "pick the blue tape and place it in the orange box.",
        )
    )
    viewer_topics_raw = data.get("viewer_topics") or []
    viewer_topics = [str(item) for item in viewer_topics_raw]
    image_throttle_hz = float(data.get("image_throttle_hz", 10.0))
    other_throttle_hz = float(data.get("other_throttle_hz", 100.0))
    enable_image_compression = bool(data.get("enable_image_compression", True))
    stop_key = str(data.get("stop_key", "q"))
    discard_key = str(data.get("discard_key", "x"))

    return CaptureConfig(
        output_root=output_root,
        task_name=task_name,
        prompt=prompt,
        viewer_topics=viewer_topics,
        image_throttle_hz=image_throttle_hz,
        other_throttle_hz=other_throttle_hz,
        enable_image_compression=enable_image_compression,
        stop_key=stop_key,
        discard_key=discard_key,
        topics=topics,
        bag=bag,
        raw=data,
        source_path=path,
    )
