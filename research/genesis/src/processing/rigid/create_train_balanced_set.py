import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


DATASET = os.environ.get("DATASET", "train_08142026")
SEED = int(os.environ.get("SEED", "42"))
GENESIS_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
DATA_DIR = Path(os.environ.get("FT_GENESIS_DATA_ROOT", GENESIS_ROOT / "data")) / DATASET
INPUT_CSV = DATA_DIR / os.environ.get("INPUT_CSV", f"{DATASET}.csv")
OUTPUT_CSV = DATA_DIR / os.environ.get("OUTPUT_CSV", f"{DATASET}_balanced.csv")
BALANCE_PLACE_SURFACES = os.environ.get("BALANCE_PLACE_SURFACES", "0").lower() in {
    "1",
    "true",
    "yes",
}
TARGET_PER_ACTION = os.environ.get("TARGET_PER_ACTION", "").strip()

ATOMIC_ACTIONS = ["grasp", "hold", "lift", "place", "press", "release", "release midair"]
BALANCE_ACTIONS = ["grasp", "hold", "lift", "place", "press", "release"]
PLACE_SURFACES = ["flat", "step", "slope", "wobbling"]


def _balance_action(action: str) -> str:
    return "release" if action == "release midair" else action


def _effective_action_counts(frame: pd.DataFrame) -> pd.Series:
    """Count every component of a connected action once."""
    counts = pd.Series(0, index=BALANCE_ACTIONS, dtype=int)
    for action, count in frame["action"].value_counts().items():
        for component in str(action).split(" then "):
            balance_action = _balance_action(component)
            if balance_action in counts.index:
                counts[balance_action] += int(count)
    return counts


def _normalize(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _category(series: pd.Series, known: set[str]) -> pd.Series:
    normalized = _normalize(series)
    return normalized.where(normalized.isin(known), "other")


def _place_surface(frame: pd.DataFrame) -> pd.Series:
    """Map actual annotated placement supports to four terrain families."""
    if "support_surface" in frame.columns:
        support = _normalize(frame["support_surface"])
    else:
        support = pd.Series("", index=frame.index, dtype=str)

    result = pd.Series("", index=frame.index, dtype=str)
    result.loc[support.str.contains("wobbl", regex=False)] = "wobbling"
    result.loc[support.str.contains("step", regex=False)] = "step"
    result.loc[support.str.contains("slope", regex=False)] = "slope"
    result.loc[
        support.str.contains("flat", regex=False)
        | support.str.contains("fixed level", regex=False)
    ] = "flat"
    if "interaction" in frame.columns:
        # An accidental drop before support contact has no observed terrain,
        # even if an older annotation file retained its intended target in the
        # support_surface column.
        result.loc[_normalize(frame["interaction"]).eq("accidental drop")] = ""
    return result


def _most_even_bounded_counts(
    total: int,
    lower: dict[str, int],
    upper: dict[str, int],
    categories: list[str] | None = None,
) -> dict[str, int]:
    """Allocate an integer total as evenly as possible within fixed bounds."""
    categories = PLACE_SURFACES if categories is None else categories
    if total < sum(lower.values()) or total > sum(upper.values()):
        raise ValueError(
            "Bounded category total is infeasible: "
            f"total={total}, lower={lower}, upper={upper}"
        )
    result = {category: int(lower[category]) for category in categories}
    remaining = total - sum(result.values())
    while remaining:
        available = [
            category
            for category in categories
            if result[category] < int(upper[category])
        ]
        if not available:
            raise RuntimeError("Ran out of place-surface capacity during allocation")
        category = min(
            available,
            key=lambda value: (result[value], categories.index(value)),
        )
        result[category] += 1
        remaining -= 1
    return result


def _select_stratum_counts(
    candidates: pd.DataFrame,
    required_by_action: dict[str, int],
    mandatory: pd.DataFrame,
    required_place_surface: dict[str, int] | None = None,
    require_all_atomic_actions: bool = False,
    required_atomic_actions: dict[str, int] | None = None,
) -> dict[tuple[str, str, str, str], int]:
    strata = (
        candidates.groupby(
            ["action", "__weight", "__interaction", "__place_surface"],
            dropna=False,
        )
        .size()
        .sort_index()
    )
    keys = list(strata.index)
    n = len(keys)
    # Keep the action target fixed at its maximum feasible value.  Missing
    # weight/interaction labels are minimized first; only then do soft
    # light/heavy and slip/stable objectives choose among equally large sets.
    global_weight_aux = n
    global_interaction_aux = n + 1
    action_weight_aux = {
        action: n + 2 + 2 * index for index, action in enumerate(BALANCE_ACTIONS)
    }
    action_interaction_aux = {
        action: n + 3 + 2 * index for index, action in enumerate(BALANCE_ACTIONS)
    }
    n_variables = n + 2 + 2 * len(BALANCE_ACTIONS)
    lower = np.zeros(n_variables)
    upper = np.r_[strata.to_numpy(dtype=float), np.inf, np.inf]
    upper = np.r_[upper, np.full(2 * len(BALANCE_ACTIONS), np.inf)]
    integrality = np.r_[
        np.ones(n, dtype=int),
        np.zeros(n_variables - n, dtype=int),
    ]

    rows = []
    lb = []
    ub = []
    for action in BALANCE_ACTIONS:
        row = np.zeros(n_variables)
        for index, key in enumerate(keys):
            if _balance_action(key[0]) == action:
                row[index] = 1.0
        required = float(required_by_action[action])
        rows.append(row)
        lb.append(required)
        ub.append(required)

    if required_place_surface is not None:
        for surface in PLACE_SURFACES:
            row = np.zeros(n_variables)
            for index, key in enumerate(keys):
                if key[0] == "place" and key[3] == surface:
                    row[index] = 1.0
            required = float(required_place_surface[surface])
            rows.append(row)
            lb.append(required)
            ub.append(required)

    if require_all_atomic_actions:
        for action in ATOMIC_ACTIONS:
            row = np.zeros(n_variables)
            for index, key in enumerate(keys):
                if key[0] == action:
                    row[index] = 1.0
            rows.append(row)
            lb.append(1.0)
            ub.append(np.inf)

    if required_atomic_actions is not None:
        for action, required in required_atomic_actions.items():
            row = np.zeros(n_variables)
            for index, key in enumerate(keys):
                if key[0] == action:
                    row[index] = 1.0
            rows.append(row)
            lb.append(float(required))
            ub.append(float(required))

    mandatory_weight_diff = int((mandatory["__weight"] == "light").sum()) - int(
        (mandatory["__weight"] == "heavy").sum()
    )
    weight_coeff = np.array(
        [1.0 if key[1] == "light" else -1.0 if key[1] == "heavy" else 0.0 for key in keys]
    )
    row = np.zeros(n_variables)
    row[:n] = weight_coeff
    row[global_weight_aux] = -1.0
    rows.append(row)
    lb.append(-np.inf)
    ub.append(float(-mandatory_weight_diff))
    row = np.zeros(n_variables)
    row[:n] = -weight_coeff
    row[global_weight_aux] = -1.0
    rows.append(row)
    lb.append(-np.inf)
    ub.append(float(mandatory_weight_diff))

    mandatory_interaction_diff = int((mandatory["__interaction"] == "slip").sum()) - int(
        (mandatory["__interaction"] == "stable").sum()
    )
    interaction_coeff = np.array(
        [1.0 if key[2] == "slip" else -1.0 if key[2] == "stable" else 0.0 for key in keys]
    )
    row = np.zeros(n_variables)
    row[:n] = interaction_coeff
    row[global_interaction_aux] = -1.0
    rows.append(row)
    lb.append(-np.inf)
    ub.append(float(-mandatory_interaction_diff))
    row = np.zeros(n_variables)
    row[:n] = -interaction_coeff
    row[global_interaction_aux] = -1.0
    rows.append(row)
    lb.append(-np.inf)
    ub.append(float(mandatory_interaction_diff))

    def mandatory_for_action(action: str) -> pd.DataFrame:
        if mandatory.empty:
            return mandatory
        return mandatory[
            mandatory["action"].map(
                lambda value: action
                in {
                    _balance_action(component)
                    for component in str(value).split(" then ")
                }
            )
        ]

    for action in BALANCE_ACTIONS:
        action_mandatory = mandatory_for_action(action)
        action_weight_diff = int(
            action_mandatory["__weight"].eq("light").sum()
        ) - int(action_mandatory["__weight"].eq("heavy").sum())
        action_weight_coeff = np.array(
            [
                1.0
                if _balance_action(key[0]) == action and key[1] == "light"
                else -1.0
                if _balance_action(key[0]) == action and key[1] == "heavy"
                else 0.0
                for key in keys
            ]
        )
        row = np.zeros(n_variables)
        row[:n] = action_weight_coeff
        row[action_weight_aux[action]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(-action_weight_diff))
        row = np.zeros(n_variables)
        row[:n] = -action_weight_coeff
        row[action_weight_aux[action]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(action_weight_diff))

        action_interaction_diff = int(
            action_mandatory["__interaction"].eq("slip").sum()
        ) - int(action_mandatory["__interaction"].eq("stable").sum())
        action_interaction_coeff = np.array(
            [
                1.0
                if _balance_action(key[0]) == action and key[2] == "slip"
                else -1.0
                if _balance_action(key[0]) == action and key[2] == "stable"
                else 0.0
                for key in keys
            ]
        )
        row = np.zeros(n_variables)
        row[:n] = action_interaction_coeff
        row[action_interaction_aux[action]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(-action_interaction_diff))
        row = np.zeros(n_variables)
        row[:n] = -action_interaction_coeff
        row[action_interaction_aux[action]] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(action_interaction_diff))

    constraint_matrix = np.stack(rows)
    constraint_lb = np.asarray(lb)
    constraint_ub = np.asarray(ub)
    missing_label_cost = np.array(
        [float(key[1] == "other") + float(key[2] == "other") for key in keys]
    )
    coverage_objective = np.r_[
        missing_label_cost,
        np.zeros(n_variables - n),
    ]
    coverage_result = milp(
        c=coverage_objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(constraint_matrix, constraint_lb, constraint_ub),
        options={"time_limit": 120.0},
    )
    if not coverage_result.success or coverage_result.x is None:
        raise RuntimeError(
            f"Failed to maximize train label coverage: {coverage_result.message}"
        )
    optimal_missing_labels = int(
        round(float(missing_label_cost @ coverage_result.x[:n]))
    )
    coverage_row = np.r_[missing_label_cost, np.zeros(n_variables - n)]
    constraint_matrix = np.vstack([constraint_matrix, coverage_row])
    constraint_lb = np.r_[constraint_lb, float(optimal_missing_labels)]
    constraint_ub = np.r_[constraint_ub, float(optimal_missing_labels)]

    balance_objective = np.zeros(n_variables)
    balance_objective[global_weight_aux] = 0.1
    balance_objective[global_interaction_aux] = 0.1
    for action in BALANCE_ACTIONS:
        balance_objective[action_weight_aux[action]] = 1.0
        balance_objective[action_interaction_aux[action]] = 1.0
    result = milp(
        c=balance_objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(constraint_matrix, constraint_lb, constraint_ub),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to optimize train sampling: {result.message}")
    selected = np.rint(result.x[:n]).astype(int)
    selection = {key: int(count) for key, count in zip(keys, selected) if count > 0}
    for action in BALANCE_ACTIONS:
        actual = sum(
            count for key, count in selection.items() if _balance_action(key[0]) == action
        )
        if actual != required_by_action[action]:
            raise RuntimeError(f"MILP count mismatch for {action}: {actual} != {required_by_action[action]}")
    print(
        "optimized label coverage / residual imbalance: "
        f"missing={optimal_missing_labels}, "
        f"weight={result.x[global_weight_aux]:.0f}, "
        f"interaction={result.x[global_interaction_aux]:.0f}"
    )
    print(
        "per-action residual imbalance: "
        + str(
            {
                action: {
                    "light-heavy": int(round(result.x[action_weight_aux[action]])),
                    "slip-stable": int(
                        round(result.x[action_interaction_aux[action]])
                    ),
                }
                for action in BALANCE_ACTIONS
            }
        )
    )
    return selection


def main() -> None:
    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    for column in ["action", "weight", "interaction"]:
        if column not in frame.columns:
            raise ValueError(f"Missing required column: {column}")
        frame[column] = _normalize(frame[column])
    frame["__weight"] = _category(frame["weight"], {"light", "heavy"})
    frame["__interaction"] = _category(frame["interaction"], {"stable", "slip"})
    frame["__place_surface"] = _place_surface(frame)

    connected = frame["action"].str.contains(" then ", regex=False)
    early_drop = frame["interaction"].eq("accidental drop")
    mandatory_mask = connected | early_drop
    mandatory = frame[mandatory_mask].copy()

    candidates = frame[
        ~mandatory_mask & frame["action"].isin(ATOMIC_ACTIONS)
    ].copy()
    candidates["__balance_action"] = candidates["action"].map(_balance_action)
    available_by_action = candidates["__balance_action"].value_counts()
    missing = [
        action for action in BALANCE_ACTIONS if int(available_by_action.get(action, 0)) == 0
    ]
    if missing:
        raise ValueError(f"No non-mandatory rows available for atomic actions: {missing}")
    # Connected actions and early drops are an unsampled mandatory pool.
    # Connected actions contribute once to each component (double counting a
    # two-action row), while release and release-midair share one balance bin.
    mandatory_by_action = _effective_action_counts(mandatory)
    maximum_by_action = mandatory_by_action.add(available_by_action, fill_value=0).astype(int)
    if TARGET_PER_ACTION:
        target_per_action = int(TARGET_PER_ACTION)
        minimum = int(mandatory_by_action.max())
        maximum = min(int(maximum_by_action[action]) for action in BALANCE_ACTIONS)
        if not minimum <= target_per_action <= maximum:
            raise ValueError(
                "TARGET_PER_ACTION is outside the feasible range: "
                f"target={target_per_action}, range=[{minimum}, {maximum}]"
            )
    elif BALANCE_PLACE_SURFACES:
        # Match terrain_floor_balanced by maximizing the common effective
        # action count.  Surface counts are then made as even as the scarce
        # wobbling-place pool and mandatory rows allow, without duplication.
        target_per_action = min(
            int(maximum_by_action[action]) for action in BALANCE_ACTIONS
        )
    else:
        target_per_action = min(int(maximum_by_action[action]) for action in BALANCE_ACTIONS)
    if target_per_action < int(mandatory_by_action.max()):
        raise ValueError(
            "Mandatory connected/early-drop action counts cannot be equalized with available rows"
        )
    required_by_action = {
        action: target_per_action - int(mandatory_by_action[action])
        for action in BALANCE_ACTIONS
    }
    required_place_surface = None
    desired_place_surface = None
    desired_release_actions = None
    if BALANCE_PLACE_SURFACES:
        mandatory_place = mandatory[
            mandatory["action"].map(
                lambda value: "place" in str(value).split(" then ")
            )
        ]
        mandatory_surface = {
            surface: int(mandatory_place["__place_surface"].eq(surface).sum())
            for surface in PLACE_SURFACES
        }
        mandatory_without_surface = int(mandatory_place["__place_surface"].eq("").sum())
        available_surface = {
            surface: int(
                (
                    candidates["action"].eq("place")
                    & candidates["__place_surface"].eq(surface)
                ).sum()
            )
            for surface in PLACE_SURFACES
        }
        maximum_surface = {
            surface: mandatory_surface[surface] + available_surface[surface]
            for surface in PLACE_SURFACES
        }
        desired_place_surface = _most_even_bounded_counts(
            total=target_per_action - mandatory_without_surface,
            lower=mandatory_surface,
            upper=maximum_surface,
        )
        required_place_surface = {
            surface: desired_place_surface[surface] - mandatory_surface[surface]
            for surface in PLACE_SURFACES
        }
        if sum(required_place_surface.values()) != required_by_action["place"]:
            raise RuntimeError(
                "Place-surface requirements do not match the required place action count"
            )
        release_actions = ["release", "release midair"]
        release_available = {
            action: int(candidates["action"].eq(action).sum())
            for action in release_actions
        }
        desired_release_actions = _most_even_bounded_counts(
            total=required_by_action["release"],
            lower={action: 1 for action in release_actions},
            upper=release_available,
            categories=release_actions,
        )

    selection = _select_stratum_counts(
        candidates,
        required_by_action,
        mandatory,
        required_place_surface=required_place_surface,
        require_all_atomic_actions=BALANCE_PLACE_SURFACES,
        required_atomic_actions=desired_release_actions,
    )
    sampled = []
    for (action, weight, interaction, place_surface), count in selection.items():
        pool = candidates[
            candidates["action"].eq(action)
            & candidates["__weight"].eq(weight)
            & candidates["__interaction"].eq(interaction)
            & candidates["__place_surface"].eq(place_surface)
        ]
        sampled.append(pool.sample(n=count, replace=False, random_state=SEED))

    output = pd.concat([mandatory, *sampled], ignore_index=True)
    output = output.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    output = output.drop(
        columns=[
            column
            for column in output.columns
            if column
            in {
                "label",
                "__weight",
                "__interaction",
                "__balance_action",
                "__place_surface",
            }
            or column.endswith("_emb_index")
        ],
        errors="ignore",
    )
    output.to_csv(OUTPUT_CSV, index=False)

    atomic_output = output[output["action"].isin(ATOMIC_ACTIONS)]
    effective_output = _effective_action_counts(output)
    print(f"saved {len(output)} rows -> {OUTPUT_CSV}")
    print(f"mandatory connected/early-drop rows: {len(mandatory)}")
    print(f"effective target per balanced action: {target_per_action}")
    print(f"action distribution: {output['action'].value_counts().sort_index().to_dict()}")
    print(f"atomic distribution: {atomic_output['action'].value_counts().sort_index().to_dict()}")
    print(f"effective action distribution: {effective_output.to_dict()}")
    if desired_place_surface is not None:
        output_surface = _place_surface(output)
        place_components = output["action"].map(
            lambda value: "place" in str(value).split(" then ")
        )
        actual_place_surface = {
            surface: int((place_components & output_surface.eq(surface)).sum())
            for surface in PLACE_SURFACES
        }
        print(f"place support target: {desired_place_surface}")
        print(f"place support distribution: {actual_place_surface}")
    if desired_release_actions is not None:
        print(f"atomic release target: {desired_release_actions}")
    print(f"weight distribution: {output['weight'].replace('', 'N/A').value_counts().to_dict()}")
    print(
        "interaction distribution: "
        f"{output['interaction'].replace('', 'N/A').value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
