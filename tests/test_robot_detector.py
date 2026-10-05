"""Exercise learned placement inference without creating a ROS node."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LEARNING_SITE = ROOT / ".venvs/learning/lib/python3.10/site-packages"
ROBOT_PACKAGE = ROOT / "robot/ros2_src/data_capture_tools"
sys.path.insert(0, str(LEARNING_SITE))
sys.path.insert(0, str(ROBOT_PACKAGE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data_capture_tools.place_cosine_detector import CosinePlaceDetector  # noqa: E402


def test_tokenfusion_place_detector_scores_one_window_on_cpu():
    patchtst = ROOT / "workspace/PatchTST"
    run = (
        "terrain_floor_wobble_uniform_descent_contact_clean_"
        "slip5mms_20260824_132352"
    )
    detector = CosinePlaceDetector(
        checkpoint_path=(
            patchtst / "saved_models/terrain_contrastive" / run
            / "terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42"
            / "terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"
        ),
        normalization_path=ROOT / "artifacts/robot/pick_place_normalization.json",
        hold_prototype_path=(
            patchtst / "results/terrain_contrastive" / run
            / "prepared_eval_templates_release_merged/bert_emb/action/1.pt"
        ),
        place_prototype_path=(
            patchtst / "results/terrain_contrastive" / run
            / "prepared_eval_templates_release_merged/bert_emb/action/3.pt"
        ),
        patchtst_code=patchtst / "PatchTST_self_supervised",
        device_name="cpu",
        expected_model_arch="tokenfusion",
    )
    scores = detector.score(np.zeros((80, 12), dtype=np.float32))
    assert detector.context_points == 80
    assert detector.normalization_sha256 == (
        "a1abf26900e7041244c635105ca791342b0a24b2df734a1afde3668ce2a368f9"
    )
    assert all(np.isfinite(value) for value in scores.values())
    assert Path(torch.__file__).resolve().is_relative_to(LEARNING_SITE.resolve())


if __name__ == "__main__":
    test_tokenfusion_place_detector_scores_one_window_on_cpu()
    print("learned placement detector: offline CPU score passed")
