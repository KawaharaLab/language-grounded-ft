"""Frozen, deterministic language interventions for the new RQ1 experiments.

This module has no filesystem, model-loading, or GPU side effects. Caption text
alone is subject to the lexical ban: metadata retains the original class names
so that supervision and positive-pair identity never depend on a paraphrase.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping


ACTION_CLASSES = ("grasp", "hold", "lift", "place", "press", "release")
WEIGHT_CLASSES = ("heavy", "light")
INTERACTION_CLASSES = ("accidental drop", "slip", "stable")
SEMANTIC_FIELDS = (
    "action", "weight", "interaction", "support_surface", "support_condition"
)

# Substrings intentionally include inflections and compounds, e.g. placing,
# released, pressure, unstable, instability, dropped, accidentally, and held.
# These evaluated word families must be absent even in a negated sentence.
FORBIDDEN_PATTERN = re.compile(
    r"grasp|releas|hold|held|lift|plac|press|slip|stabl|stabil|drop|acciden",
    re.ASCII | re.IGNORECASE,
)

_SURFACES = {
    "", "a fixed flat surface", "a flat surface", "a gentle slope",
    "a moderate slope", "a steep slope", "a small step", "a medium step",
    "a large step", "a wobbling surface",
}

# Eight physical descriptions per action, rather than 32 adjective swaps.
# Infinitives are converted to gerunds before sentence assembly. The explicit
# article-bearing {obj} is the only location at which known weight is inserted.
_ACTION_BASES = {
    "grasp": (
        "close the fingers around {obj}",
        "bring both fingers inward to grip {obj}",
        "move the fingers toward {obj} to form a grip",
        "narrow the gap between the fingers around {obj}",
        "form a grip around {obj} by closing the fingers",
        "bring the fingertips together against {obj}",
        "draw the fingers inward around {obj}",
        "make contact with {obj} as the fingers close",
    ),
    "lift": (
        "raise {obj} from its support",
        "move {obj} upward and away from its support",
        "carry {obj} upward from the supporting surface",
        "bring {obj} upward until it leaves its support",
        "move {obj} off the support in an upward direction",
        "raise {obj} clear of the surface beneath it",
        "bring {obj} away from its support by moving upward",
        "raise {obj} into the air from the supporting surface",
    ),
    "hold": (
        "keep {obj} between the fingers",
        "retain {obj} in the gripper while it is clear of support",
        "keep {obj} suspended in the gripper",
        "maintain a grip on {obj} away from any supporting surface",
        "retain {obj} between the fingers in midair",
        "keep {obj} in the gripper above its support",
        "support {obj} using the fingers while it is in the air",
        "maintain {obj} between the fingers while it remains suspended",
    ),
    "place": (
        "bring {obj} into contact with {surface}",
        "move {obj} down until it meets {surface}",
        "lower {obj} to make contact with {surface}",
        "bring {obj} down onto {surface}",
        "move {obj} downward onto {surface}",
        "set {obj} down on {surface}",
        "guide {obj} downward until it touches {surface}",
        "move {obj} into contact with {surface} from above",
    ),
    "press": (
        "apply downward force to {obj} against {surface}",
        "push {obj} downward while it is already touching {surface}",
        "drive {obj} downward against {surface} after contact",
        "continue pushing {obj} into {surface} after contact",
        "exert downward force on {obj} while it touches {surface}",
        "push down on {obj} against {surface} after contact",
        "continue downward force on {obj} already touching {surface}",
        "push {obj} against {surface} after it has made contact",
    ),
    "release": (
        "open the fingers to let go of {obj} on the supporting surface",
        "move the fingers apart to let go of {obj} resting on a surface",
        "open the gripper so that {obj} remains on its support",
        "end the grip on {obj} while it rests on a supporting surface",
        "separate the fingers to let go of {obj} on its support",
        "widen the finger gap to let go of {obj} resting on a surface",
        "open both fingers away from {obj} resting on its support",
        "let go of {obj} by opening the fingers while it rests on a surface",
    ),
    "release midair": (
        "open the fingers to let go of {obj} in midair",
        "move the fingers apart to let go of {obj} while it is in the air",
        "open the gripper to let go of {obj} away from any supporting surface",
        "end the grip on {obj} while it is in midair",
        "separate the fingers to let go of {obj} before it reaches any support",
        "widen the finger gap to let go of {obj} in the air",
        "open both fingers away from {obj} while it is in midair",
        "let go of {obj} in the air by opening the fingers",
    ),
}

_GERUNDS = {
    "apply": "applying", "bring": "bringing", "carry": "carrying",
    "close": "closing", "continue": "continuing", "draw": "drawing",
    "drive": "driving", "end": "ending", "exert": "exerting",
    "form": "forming", "guide": "guiding", "keep": "keeping",
    "let": "letting", "lower": "lowering", "maintain": "maintaining",
    "make": "making", "move": "moving", "narrow": "narrowing",
    "open": "opening", "push": "pushing", "raise": "raising",
    "retain": "retaining", "separate": "separating", "set": "setting",
    "support": "supporting", "widen": "widening",
}

# Positive wording for the sub-threshold state deliberately does not claim
# exactly zero relative motion. These are categorical paraphrases of the saved
# annotation, not a new displacement/velocity rule. For a sequence ending in
# opening the fingers, the interaction describes its finger-contact portion.
_INTERACTION_CLAUSES = {
    "stable": (
        "The object has secure contact with the fingers",
        "The object stays steady in the grip during finger contact",
        "The grip on the object is steady during finger contact",
        "The contact between the object and the fingers is secure",
    ),
    "slip": (
        "The object slides relative to the fingers during contact",
        "The object is sliding within the grip during finger contact",
        "There is tangential sliding between the object and the fingers",
        "The object moves along the finger surfaces during contact",
    ),
    "accidental drop": (
        "The object falls unintentionally out of the fingers",
        "The object unintentionally falls out of the grip",
        "The gripper loses the object unintentionally and it falls",
        "The object falls from the gripper unintentionally",
    ),
}


def _text(value: Any) -> str:
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value)
    return "" if value in ("nan", "<NA>", "None") else value.strip()


def _records(frame: Any) -> Iterable[Mapping[str, Any]]:
    """Accept a pandas DataFrame or a record iterable without importing pandas."""
    if hasattr(frame, "to_dict") and hasattr(frame, "columns"):
        columns = ["emb_index", "annotation", *SEMANTIC_FIELDS]
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing caption columns: {sorted(missing)}")
        yield from frame[columns].drop_duplicates().to_dict(orient="records")
    else:
        yield from frame


def _gerund(phrase: str) -> str:
    first, rest = phrase.split(" ", 1)
    return _GERUNDS[first] + " " + rest


def _capitalized(text: str) -> str:
    return text[0].upper() + text[1:]


def _action_phrase(state: Mapping[str, str], variant: int, gerund: bool) -> str:
    action = state["action"]
    components = action.split(" then ")
    obj = f"a {state['weight']} object" if state["weight"] else "an object"
    surface = state["support_surface"] or "a supporting surface"
    phrases = []
    for index, component in enumerate(components):
        phrase = _ACTION_BASES[component][variant].format(
            obj=obj if index == 0 else "it", surface=surface
        )
        phrases.append(_gerund(phrase) if gerund else phrase)
    return " and then ".join(phrases)


def _caption(state: Mapping[str, str], action_variant: int, style: int) -> str:
    interaction = state["interaction"]
    if interaction == "accidental drop":
        # The action label is the intended action during failure. Do not turn
        # an unsuccessful raising/setting-down attempt into a claimed success.
        intended = _action_phrase(state, action_variant, gerund=False)
        failure = _INTERACTION_CLAUSES[interaction][style]
        if style == 0:
            return f"The gripper tries to {intended}. {failure}."
        if style == 1:
            return f"While trying to {intended}, {failure[0].lower() + failure[1:]}."
        if style == 2:
            return f"The robot is attempting to {intended}. {failure}."
        return f"{failure}. This happens while the gripper is trying to {intended}."

    action = _action_phrase(state, action_variant, gerund=True)
    clause = _INTERACTION_CLAUSES[interaction][style] if interaction else ""
    if state["action"].endswith(" then release") and clause:
        # Opening eventually ends finger contact. Do not invent a finer event
        # order: relative motion can also occur while the fingers are opening.
        clause = "In the portion with finger contact, " + clause[0].lower() + clause[1:]
    if style == 0:
        return _capitalized(action) + "." + (f" {clause}." if clause else "")
    if style == 1:
        return f"The gripper is {action}." + (f" {clause}." if clause else "")
    if style == 2:
        return (f"{clause}. " if clause else "") + f"The robot is {action}."
    text = f"This action involves {action}"
    return text + (f"; {clause[0].lower() + clause[1:]}" if clause else "") + "."


def build_caption_bank(frame: Any) -> dict[str, Any]:
    """Build three aligned banks indexed by the source's contiguous emb_index.

    canonical is copied byte-for-byte at the string level from annotation.
    single is one predeclared non-overlapping-vocabulary alternative per ID.
    diverse is eight action descriptions by four sentence structures per ID;
    variant zero equals single. No physical windows are duplicated or filtered.
    """
    states: dict[int, dict[str, str]] = {}
    canonical: dict[int, str] = {}
    for row in _records(frame):
        raw_index = row["emb_index"]
        index = int(raw_index)
        if float(raw_index) != index or index < 0:
            raise ValueError(f"Invalid emb_index: {raw_index!r}")
        state = {key: _text(row[key]) for key in SEMANTIC_FIELDS}
        if state["action"] not in {
            *_ACTION_BASES, "grasp then lift", "place then release", "press then release"
        }:
            raise ValueError(f"Unsupported action: {state['action']!r}")
        if state["weight"] not in {"", *WEIGHT_CLASSES}:
            raise ValueError(f"Unsupported weight: {state['weight']!r}")
        if state["interaction"] not in {"", *INTERACTION_CLASSES}:
            raise ValueError(f"Unsupported interaction: {state['interaction']!r}")
        if state["support_surface"] not in _SURFACES:
            raise ValueError(f"Unsupported support_surface: {state['support_surface']!r}")
        if state["support_surface"] and not any(
            component in ("place", "press") for component in state["action"].split(" then ")
        ):
            raise ValueError("A surface attribute cannot be silently lost on this action")
        if state["support_condition"] not in {"", "fixed support", "wobbling support"}:
            raise ValueError(f"Unsupported support condition: {state['support_condition']!r}")
        if state["support_condition"] == "fixed support" and state["support_surface"] != "a fixed flat surface":
            raise ValueError("Fixed support is not represented by the surface phrase")
        if state["support_condition"] == "wobbling support" and state["support_surface"] != "a wobbling surface":
            raise ValueError("Moving support is not represented by the surface phrase")
        annotation = row["annotation"]
        if not isinstance(annotation, str) or not annotation:
            raise ValueError(f"Missing canonical annotation for {index}")
        if index in states and (states[index] != state or canonical[index] != annotation):
            raise ValueError(f"emb_index {index} refers to multiple semantic states/texts")
        states[index] = state
        canonical[index] = annotation

    expected = list(range(len(states)))
    if sorted(states) != expected:
        raise ValueError("Use the complete source index map; emb_index must be contiguous from zero")
    if not states:
        raise ValueError("The caption source is empty")

    diverse = [
        [_caption(states[index], action_variant, style)
         for action_variant in range(8) for style in range(4)]
        for index in expected
    ]
    bank = {
        "canonical": [canonical[index] for index in expected],
        "single": [variants[0] for variants in diverse],
        "diverse": diverse,
        "metadata": {
            "version": 1,
            "num_semantic_ids": len(states),
            "variants_per_id": 32,
            "variant_order": "action_description_index * 4 + sentence_structure_index",
            "semantic_states": [states[index] for index in expected],
            "semantic_id": "Original emb_index, invariant across all caption variants",
            "forbidden_regex": FORBIDDEN_PATTERN.pattern,
            "ban_scope": "Every full single/diverse caption, including support descriptions",
            "stability_meaning": "Saved sub-5 mm/s interaction label; secure/steady wording does not assert exactly zero motion",
            "weight_policy": "Preserve heavy/light verbatim; do not invent unknown weight",
            "surface_policy": "Preserve exact surface phrase if annotated; no surface labels before surface contact actions",
            "diversity_policy": "Eight reviewed physical descriptions and four sentence structures; no model/test-score selection",
            "positive_identity": "All variants sharing emb_index have the same semantic target",
        },
    }
    bank["metadata"]["audit"] = audit_caption_bank(bank)
    return bank


def audit_caption_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on lexical leakage, duplicate variants, or changed attributes."""
    n = len(bank["canonical"])
    if len(bank["single"]) != n or len(bank["diverse"]) != n:
        raise ValueError("Caption banks have different numbers of semantic IDs")
    states = bank["metadata"]["semantic_states"]
    seen: dict[str, int] = {}
    violations = []
    for index in range(n):
        variants = bank["diverse"][index]
        if len(variants) != 32 or len(set(variants)) != 32:
            raise ValueError(f"Semantic ID {index} does not have 32 unique variants")
        if bank["single"][index] != variants[0]:
            raise ValueError(f"Single caption is not variant zero for {index}")
        state = states[index]
        for variant, caption in enumerate(variants):
            if not caption.isascii():
                raise ValueError("Caption lexical audit expects ASCII; Unicode must not bypass the ban")
            matches = [m.group(0) for m in FORBIDDEN_PATTERN.finditer(caption.casefold())]
            if matches:
                violations.append({"id": index, "variant": variant, "matches": matches})
            weight_words = re.findall(r"\b(?:heavy|light)\b", caption)
            if weight_words != ([state["weight"]] if state["weight"] else []):
                raise ValueError(f"Caption changes or duplicates weight at ID {index}/{variant}")
            if state["support_surface"] and state["support_surface"] not in caption:
                raise ValueError(f"Caption loses the original surface at ID {index}/{variant}")
            if caption in seen and seen[caption] != index:
                raise ValueError(f"Different IDs {seen[caption]} and {index} have the same paraphrase")
            seen[caption] = index
    if violations:
        raise ValueError(f"Forbidden vocabulary detected: {violations[:10]}")
    payload = {key: bank[key] for key in ("canonical", "single", "diverse")}
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "checked_single_captions": n,
        "checked_diverse_captions": n * 32,
        "forbidden_match_count": 0,
        "unique_diverse_captions": len(seen),
        "duplicate_within_id_count": 0,
        "weight_mismatch_count": 0,
        "surface_loss_count": 0,
        "canonical_bank_sha256": hashlib.sha256(
            json.dumps(bank["canonical"], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "all_caption_banks_sha256": digest,
        "semantic_action_counts": dict(Counter(s["action"] for s in states)),
    }


def build_eval_suites() -> dict[str, dict[str, dict[str, str]]]:
    """Return frozen main, bare-word, and five new held-out prompt suites.

    Weight wording is held fixed in the five paraphrase suites because the
    linguistic intervention concerns action and interaction. No ensembling or
    best-suite selection is implied. Canonical wording is the actual current
    RQ1 wording, including 'stable contact', rather than an older prototype.
    """
    weights = {"heavy": "The object is heavy.", "light": "The object is light."}
    canonical = {
        "action": dict(zip(ACTION_CLASSES, (
            "Grasping an object.", "Holding an object.", "Lifting an object.",
            "Placing an object.", "Pressing an object.", "Releasing an object.",
        ))),
        "weight": dict(weights),
        "interaction": dict(zip(INTERACTION_CLASSES, (
            "The object was accidentally dropped.", "The object is slipping.",
            "The object has stable contact.",
        ))),
    }
    bare = {
        "action": {name: name for name in ACTION_CLASSES},
        "weight": {name: name for name in WEIGHT_CLASSES},
        "interaction": {name: name for name in INTERACTION_CLASSES},
    }
    actions = (
        (
            "An object is being grasped by the robot.",
            "The robot is holding an object in its fingers.",
            "The robot is lifting an object away from its support.",
            "The robot is placing an object onto a support.",
            "The robot is pressing an object downward against a surface.",
            "The robot is releasing its grip on an object.",
        ),
        (
            "Grasping an object between two fingers.",
            "Holding an object above a supporting surface.",
            "Lifting an object clear of the surface beneath it.",
            "Placing an object so it meets a supporting surface.",
            "Pressing downward on an object that is already on a surface.",
            "Releasing an object by opening the fingers.",
        ),
        (
            "The fingers close inward to take an object into their grip.",
            "An object remains suspended between the robot's fingers.",
            "An object travels upward as it leaves the supporting surface.",
            "An object reaches the supporting surface as the robot moves it down.",
            "An object receives a downward push after touching a support.",
            "An object is let go when the robot spreads its fingers.",
        ),
        (
            "Closing a grip around an object with the robot's fingers.",
            "Keeping an object suspended within the robot's grip.",
            "Raising an object away from the surface it was resting on.",
            "Setting an object down until it touches the surface beneath it.",
            "Applying a downward push through an object onto a support.",
            "Opening the robot's fingers and letting an object go.",
        ),
        (
            "The action is to establish a grasp around an object.",
            "The action is to hold an object in the air.",
            "The action is to lift an object off its support.",
            "The action is to place an object in contact with a surface.",
            "The action is to press an object against its support.",
            "The action is to release the object from the fingers.",
        ),
    )
    interactions = (
        (
            "An object is accidentally dropped from the robot's fingers.",
            "The object is slipping relative to the robot's fingers.",
            "Contact with the object remains stable.",
        ),
        (
            "The gripper accidentally drops the object during the action.",
            "The object slips along the fingers during contact.",
            "The gripper has stable contact with the object.",
        ),
        (
            "The robot loses the object unintentionally and the object falls.",
            "An object slides against the fingertips that are touching it.",
            "The object stays secure in contact with the fingertips.",
        ),
        (
            "The object unintentionally falls free of the robot's grip.",
            "The object has sliding motion relative to its fingertip contacts.",
            "The grip on the object remains steady and secure.",
        ),
        (
            "The interaction involves an accidental drop from the gripper.",
            "The interaction involves slip between the object and the fingers.",
            "The interaction involves stable object-finger contact.",
        ),
    )
    suites = {"canonical": canonical, "bare_words": bare}
    for index, (action_texts, interaction_texts) in enumerate(zip(actions, interactions), 1):
        suites[f"paraphrase_{index}"] = {
            "action": dict(zip(ACTION_CLASSES, action_texts)),
            "weight": dict(weights),
            "interaction": dict(zip(INTERACTION_CLASSES, interaction_texts)),
        }
    return suites


def build_composition_prototypes() -> dict[str, str]:
    """Twelve joint candidates, action-major and heavy/light-minor, with '|'."""
    verbs = ("Grasping", "Holding", "Lifting", "Placing", "Pressing", "Releasing")
    return {
        f"{action}|{weight}": f"{verb} a {weight} object."
        for action, verb in zip(ACTION_CLASSES, verbs)
        for weight in WEIGHT_CLASSES
    }
