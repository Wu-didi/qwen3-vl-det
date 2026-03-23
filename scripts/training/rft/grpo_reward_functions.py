#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reward functions for Stage-1 traffic-device GRPO training.

Expected dataset columns per sample:
    - prompt: str
    - image: str (not used directly here, but kept by dataset)
    - ground_truth: {"detections": [ ... ]}
    - difficulty: {
          "num_targets": int,
          "num_abnormal": int,
          "num_device_types": int,
          "min_box_area": float | None,
          "min_short_side": float | None,
          "has_tiny_object": bool,
          "has_ignore_region": bool,
          "has_distractor": bool,
          "empty_scene": bool,
      }

Each detection is expected to be:
    {
        "device_type": str,
        "sub_type": str | None,
        "state": str,
        "bbox_1000": [x1, y1, x2, y2]
    }

The reward functions follow TRL GRPO's custom reward convention:
    def reward_func(prompts, completions, ground_truth, difficulty, **kwargs) -> list[float]

These functions are designed for standard-format outputs where the model completion
is a string. They also tolerate conversational-format completions
(list[dict(role, content)]).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ALLOWED_DEVICE_TYPES = {
    "traffic_signal",
    "guidance_screen",
    "height_limit_bar",
    "cabinet",
    "backpack_box",
}

ALLOWED_SUB_TYPES = {
    "vehicle_signal",
    "pedestrian_signal",
    None,
}

ALLOWED_STATES_BY_TYPE = {
    "traffic_signal": {"normal", "all-off", "all-on", "abnormal"},
    "guidance_screen": {"normal", "black-screen", "abnormal"},
    "height_limit_bar": {"normal", "abnormal"},
    "cabinet": {"normal", "abnormal"},
    "backpack_box": {"normal", "abnormal"},
}


# ----------------------------
# Parsing helpers
# ----------------------------

def _completion_to_text(completion: Any) -> str:
    """Handle both standard-format strings and conversational-format completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL conversational format often passes a one-message completion.
        texts: List[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                texts.append(str(item["content"]))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(completion)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to recover the first top-level JSON object from free-form text."""
    text = text.strip()
    if not text:
        return None

    # Fast path: full string is valid JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Recover from extra prose by scanning balanced braces.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None


def _normalize_scalar_label(value: Any) -> Optional[str]:
    """Collapse singleton list/tuple wrappers and normalize scalar labels."""
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
    else:
        value = str(value).strip()

    if value == "" or value.lower() == "null":
        return None
    return value


def _normalize_sub_type(value: Any) -> Optional[str]:
    return _normalize_scalar_label(value)


def _normalize_bbox(bbox: Any) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if min(x1, y1, x2, y2) < 0 or max(x1, y1, x2, y2) > 1000:
        return None
    return [x1, y1, x2, y2]


def _parse_prediction(text: str) -> Optional[List[Dict[str, Any]]]:
    obj = _extract_json_object(text)
    if obj is None:
        return None
    detections = obj.get("detections")
    if not isinstance(detections, list):
        return None

    parsed: List[Dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            return None
        device_type = _normalize_scalar_label(det.get("device_type"))
        sub_type = _normalize_sub_type(det.get("sub_type"))
        state = _normalize_scalar_label(det.get("state"))
        bbox_1000 = _normalize_bbox(det.get("bbox_1000"))
        parsed.append(
            {
                "device_type": device_type,
                "sub_type": sub_type,
                "state": state,
                "bbox_1000": bbox_1000,
            }
        )
    return parsed


# ----------------------------
# Validation helpers
# ----------------------------

def _is_valid_detection(det: Dict[str, Any]) -> bool:
    device_type = det.get("device_type")
    sub_type = det.get("sub_type")
    state = det.get("state")
    bbox = det.get("bbox_1000")

    if not isinstance(device_type, str):
        return False
    if device_type not in ALLOWED_DEVICE_TYPES:
        return False
    if sub_type not in ALLOWED_SUB_TYPES:
        return False
    if not isinstance(state, str):
        return False
    if state not in ALLOWED_STATES_BY_TYPE.get(device_type, set()):
        return False
    if bbox is None:
        return False
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
        return False
    # subtype consistency: only traffic_signal should normally have a subtype.
    if device_type != "traffic_signal" and sub_type is not None:
        return False
    return True


# ----------------------------
# Matching helpers
# ----------------------------

def _iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _label_compatible(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    if pred["device_type"] != gt["device_type"]:
        return False
    # For traffic signals, subtype matters.
    if pred["device_type"] == "traffic_signal":
        return pred.get("sub_type") == gt.get("sub_type")
    return True


def _eval_aligned_label_compatible(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    """Match the offline evaluator: device_type + IoU only."""
    return pred.get("device_type") == gt.get("device_type")


def _match_predictions(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    label_compatible_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy one-to-one matching under label compatibility and IoU threshold.
    Returns:
        matches: list of (pred_idx, gt_idx, iou)
        unmatched_pred_indices
        unmatched_gt_indices
    """
    candidates: List[Tuple[float, int, int]] = []
    label_compatible_fn = label_compatible_fn or _label_compatible
    for pi, pred in enumerate(preds):
        if pred.get("bbox_1000") is None:
            continue
        for gi, gt in enumerate(gts):
            if not label_compatible_fn(pred, gt):
                continue
            iou = _iou_xyxy(pred["bbox_1000"], gt["bbox_1000"])
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    # Highest IoU first.
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    unmatched_preds = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gts = [i for i in range(len(gts)) if i not in used_gt]
    return matches, unmatched_preds, unmatched_gts


def _safe_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def _category_key(det: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Offline evaluator only groups JSON detections by device_type."""
    device_type = det.get("device_type")
    return device_type, None


# ----------------------------
# Difficulty weighting
# ----------------------------

def _difficulty_multiplier(difficulty: Optional[Dict[str, Any]]) -> float:
    if not isinstance(difficulty, dict):
        return 1.0
    w = 1.0
    if difficulty.get("has_tiny_object"):
        w += 0.15
    if difficulty.get("num_abnormal", 0) > 0:
        w += 0.10
    if difficulty.get("empty_scene"):
        w += 0.10
    if difficulty.get("has_ignore_region"):
        w += 0.05
    return w


# ----------------------------
# Reward components
# ----------------------------

def format_reward_func(completions, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        rewards.append(0.2 if parsed is not None else 0.0)
    return rewards



def schema_reward_func(completions, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        if parsed is None:
            rewards.append(0.0)
            continue
        if len(parsed) == 0:
            rewards.append(0.2)
            continue
        valid_count = sum(1 for det in parsed if _is_valid_detection(det))
        rewards.append(0.2 * (valid_count / max(1, len(parsed))))
    return rewards



def detection_reward_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt, diff in zip(completions, ground_truth, difficulty or [None] * len(completions)):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        matches, unmatched_preds, unmatched_gts = _match_predictions(preds_valid, gts, iou_threshold=0.5)

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        rewards.append(0.6 * f1 * _difficulty_multiplier(diff))
    return rewards



def state_reward_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt, diff in zip(completions, ground_truth, difficulty or [None] * len(completions)):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        matches, _, _ = _match_predictions(preds_valid, gts, iou_threshold=0.5)
        if not matches:
            rewards.append(0.0)
            continue
        correct = 0
        for pi, gi, _ in matches:
            if preds_valid[pi]["state"] == gts[gi]["state"]:
                correct += 1
        acc = correct / len(matches)
        rewards.append(0.25 * acc * _difficulty_multiplier(diff))
    return rewards



def bbox_reward_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt, diff in zip(completions, ground_truth, difficulty or [None] * len(completions)):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        matches, _, _ = _match_predictions(preds_valid, gts, iou_threshold=0.5)
        if not matches:
            rewards.append(0.0)
            continue
        mean_iou = sum(iou for _, _, iou in matches) / len(matches)
        rewards.append(0.25 * mean_iou * _difficulty_multiplier(diff))
    return rewards



def empty_scene_reward_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt, diff in zip(completions, ground_truth, difficulty or [None] * len(completions)):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        gt_empty = len(gts) == 0

        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]
        pred_empty = len(preds_valid) == 0

        if gt_empty and pred_empty:
            rewards.append(0.40 * _difficulty_multiplier(diff))
        elif gt_empty and not pred_empty:
            rewards.append(-0.30 * _difficulty_multiplier(diff))
        else:
            rewards.append(0.0)
    return rewards



def duplicate_penalty_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    """Penalty for unmatched valid predictions after one-to-one matching."""
    rewards: List[float] = []
    for completion, gt, diff in zip(completions, ground_truth, difficulty or [None] * len(completions)):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        _, unmatched_preds, _ = _match_predictions(preds_valid, gts, iou_threshold=0.5)
        penalty = -0.05 * len(unmatched_preds)
        rewards.append(penalty)
    return rewards


# ----------------------------
# Combined reward (single function)
# ----------------------------

def combined_reward_func(completions, ground_truth, difficulty=None, **kwargs) -> List[float]:
    """
    Single combined reward function.

    Recommended if you prefer one reward function in GRPOTrainer.
    If you prefer clearer logging / ablation, pass the individual reward
    functions as a list to GRPOTrainer and optionally provide reward_weights.
    """
    r_format = format_reward_func(completions=completions, **kwargs)
    r_schema = schema_reward_func(completions=completions, **kwargs)
    r_det = detection_reward_func(completions=completions, ground_truth=ground_truth, difficulty=difficulty, **kwargs)
    r_state = state_reward_func(completions=completions, ground_truth=ground_truth, difficulty=difficulty, **kwargs)
    r_bbox = bbox_reward_func(completions=completions, ground_truth=ground_truth, difficulty=difficulty, **kwargs)
    r_empty = empty_scene_reward_func(completions=completions, ground_truth=ground_truth, difficulty=difficulty, **kwargs)
    r_dup = duplicate_penalty_func(completions=completions, ground_truth=ground_truth, difficulty=difficulty, **kwargs)

    rewards: List[float] = []
    for vals in zip(r_format, r_schema, r_det, r_state, r_bbox, r_empty, r_dup):
        total = sum(vals)
        rewards.append(float(total))
    return rewards


# ----------------------------
# Convenience helpers
# ----------------------------

def build_reward_funcs(as_list: bool = True):
    """
    Returns either:
      - a list of reward functions for TRL GRPOTrainer, or
      - the single combined reward function.

    Example:
        trainer = GRPOTrainer(
            ...,
            reward_funcs=build_reward_funcs(as_list=True),
            reward_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    """
    funcs = [
        format_reward_func,
        schema_reward_func,
        detection_reward_func,
        state_reward_func,
        bbox_reward_func,
        empty_scene_reward_func,
        duplicate_penalty_func,
    ]
    return funcs if as_list else combined_reward_func


def parse_model_output(text: str) -> Optional[List[Dict[str, Any]]]:
    """Public helper for offline debugging."""
    return _parse_prediction(text)


# ============================================================
# V2 reward scheme – 4 orthogonal functions, all in [0, 1]
# ============================================================

def v2_format_reward(completions, ground_truth, **kwargs) -> List[float]:
    """Soft format score: JSON parseable +0.5, valid detections +0.3, reasonable count +0.2.

    Parse failure with <box> tags → 0.15 (preserve gradient).
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []
        parsed = _parse_prediction(text)

        if parsed is None:
            # Partial credit if <box> tags present (keeps gradient alive).
            if re.search(r"<box>", text):
                rewards.append(0.15)
            else:
                rewards.append(0.0)
            continue

        score = 0.5  # JSON parseable

        # Valid detection structure: all items have required fields.
        if len(parsed) == 0:
            # Empty list is structurally valid.
            score += 0.3
        else:
            valid_count = sum(1 for det in parsed if _is_valid_detection(det))
            score += 0.3 * (valid_count / len(parsed))

        # Reasonable count: not wildly off from GT.
        gt_count = len(gts)
        pred_count = len(parsed)
        if gt_count == 0:
            # Empty scene: fewer predictions → better.
            score += 0.2 if pred_count <= 2 else 0.1
        else:
            ratio = pred_count / max(gt_count, 1)
            if 0.5 <= ratio <= 2.0:
                score += 0.2
            elif 0.25 <= ratio <= 3.0:
                score += 0.1

        rewards.append(min(1.0, score))
    return rewards


def v2_detection_f1_reward(completions, ground_truth, **kwargs) -> List[float]:
    """Detection F1 (80%) + mean IoU quality (20%).

    Uses IoU threshold 0.3 and exact label matching via _label_compatible().
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]

        # Both empty → perfect.
        if len(gts) == 0 and len(preds_valid) == 0:
            rewards.append(1.0)
            continue
        # GT empty but predictions present → 0.
        if len(gts) == 0:
            rewards.append(0.0)
            continue
        # GT present but no valid predictions → 0.
        if len(preds_valid) == 0:
            rewards.append(0.0)
            continue

        matches, unmatched_preds, unmatched_gts = _match_predictions(
            preds_valid, gts, iou_threshold=0.3,
        )
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0

        mean_iou = sum(iou for _, _, iou in matches) / len(matches) if matches else 0.0

        score = 0.8 * f1 + 0.2 * mean_iou
        rewards.append(min(1.0, score))
    return rewards


def v2_state_reward(completions, ground_truth, **kwargs) -> List[float]:
    """State accuracy on matched pairs: 70% overall + 30% abnormal accuracy."""
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]

        if len(gts) == 0 and len(preds_valid) == 0:
            rewards.append(1.0)
            continue
        if len(gts) == 0:
            rewards.append(0.0)
            continue

        matches, _, _ = _match_predictions(preds_valid, gts, iou_threshold=0.3)
        if not matches:
            rewards.append(0.0)
            continue

        correct_total = 0
        correct_abnormal = 0
        abnormal_count = 0
        for pi, gi, _ in matches:
            pred_state = preds_valid[pi]["state"]
            gt_state = gts[gi]["state"]
            if pred_state == gt_state:
                correct_total += 1
            if gt_state != "normal":
                abnormal_count += 1
                if pred_state == gt_state:
                    correct_abnormal += 1

        overall_acc = correct_total / len(matches)
        abnormal_acc = (correct_abnormal / abnormal_count) if abnormal_count > 0 else 1.0

        score = 0.7 * overall_acc + 0.3 * abnormal_acc
        rewards.append(min(1.0, score))
    return rewards


def v2_coverage_reward(completions, ground_truth, **kwargs) -> List[float]:
    """Category coverage (40%) + instance recall (40%) + recall balance (20%).

    Directly addresses selective category skipping (e.g. backpack_box).
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        preds = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if preds is None:
            rewards.append(0.0)
            continue

        preds_valid = [p for p in preds if _is_valid_detection(p)]

        if len(gts) == 0 and len(preds_valid) == 0:
            rewards.append(1.0)
            continue
        if len(gts) == 0:
            rewards.append(0.0)
            continue
        if len(preds_valid) == 0:
            rewards.append(0.0)
            continue

        # -- 40%: Category type coverage --
        gt_types = set()
        for g in gts:
            key = g["device_type"]
            if g["device_type"] == "traffic_signal" and g.get("sub_type"):
                key = f"{g['device_type']}_{g['sub_type']}"
            gt_types.add(key)

        pred_types = set()
        for p in preds_valid:
            key = p["device_type"]
            if p["device_type"] == "traffic_signal" and p.get("sub_type"):
                key = f"{p['device_type']}_{p['sub_type']}"
            pred_types.add(key)

        type_coverage = len(gt_types & pred_types) / len(gt_types) if gt_types else 1.0

        # -- 40%: Instance recall (loose IoU=0.2) --
        matches_loose, _, unmatched_gts = _match_predictions(
            preds_valid, gts, iou_threshold=0.2,
        )
        instance_recall = len(matches_loose) / len(gts) if gts else 1.0

        # -- 20%: Recall balance across categories --
        # Compute per-category recall; min recall penalizes selective skipping.
        cat_gt_count: Dict[str, int] = {}
        cat_matched_count: Dict[str, int] = {}
        for g in gts:
            key = g["device_type"]
            cat_gt_count[key] = cat_gt_count.get(key, 0) + 1
            cat_matched_count.setdefault(key, 0)

        matched_gt_indices = {gi for _, gi, _ in matches_loose}
        for gi in matched_gt_indices:
            key = gts[gi]["device_type"]
            cat_matched_count[key] = cat_matched_count.get(key, 0) + 1

        per_cat_recalls = []
        for cat_key in cat_gt_count:
            cat_total = cat_gt_count[cat_key]
            cat_matched = cat_matched_count.get(cat_key, 0)
            per_cat_recalls.append(cat_matched / cat_total if cat_total > 0 else 0.0)

        min_recall = min(per_cat_recalls) if per_cat_recalls else 0.0

        score = 0.4 * type_coverage + 0.4 * instance_recall + 0.2 * min_recall
        rewards.append(min(1.0, score))
    return rewards


def simple_format_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Reward valid JSON structure while avoiding positive credit for empty positive samples.

    This keeps a small format incentive but stops the old failure mode where
    `{"detections":[]}` on a positive scene still produced a meaningful reward.
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if parsed is None:
            rewards.append(0.0)
            continue

        if len(parsed) == 0:
            rewards.append(1.0 if len(gts) == 0 else 0.0)
            continue

        valid_count = sum(1 for det in parsed if _is_valid_detection(det))
        if len(gts) > 0 and valid_count == 0:
            rewards.append(0.0)
            continue

        rewards.append(valid_count / len(parsed))
    return rewards


def simple_detection_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Main optimization target: instance F2 at IoU=0.5 with a small IoU quality term.

    F2 gives recall more weight than precision, which is a better fit for the
    current failure mode (systematically skipping hard classes / small objects).
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if parsed is None:
            rewards.append(0.0)
            continue

        preds_valid = [pred for pred in parsed if _is_valid_detection(pred)]

        if len(gts) == 0:
            rewards.append(1.0 if len(preds_valid) == 0 else 0.0)
            continue
        if len(preds_valid) == 0:
            rewards.append(0.0)
            continue

        matches, unmatched_preds, unmatched_gts = _match_predictions(
            preds_valid,
            gts,
            iou_threshold=0.5,
            label_compatible_fn=_eval_aligned_label_compatible,
        )
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2 = _safe_fbeta(precision, recall, beta=2.0)
        mean_iou = sum(iou for _, _, iou in matches) / len(matches) if matches else 0.0
        rewards.append(min(1.0, 0.85 * f2 + 0.15 * mean_iou))
    return rewards


def simple_state_reward(completions, ground_truth, **kwargs) -> List[float]:
    """State accuracy on IoU=0.5 matched pairs only."""
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if parsed is None:
            rewards.append(0.0)
            continue

        preds_valid = [pred for pred in parsed if _is_valid_detection(pred)]

        if len(gts) == 0:
            rewards.append(0.0)
            continue

        matches, _, _ = _match_predictions(
            preds_valid,
            gts,
            iou_threshold=0.5,
            label_compatible_fn=_eval_aligned_label_compatible,
        )
        if not matches:
            rewards.append(0.0)
            continue

        correct = 0
        for pred_idx, gt_idx, _ in matches:
            if preds_valid[pred_idx]["state"] == gts[gt_idx]["state"]:
                correct += 1
        rewards.append(correct / len(matches))
    return rewards


def simple_category_recall_reward(completions, ground_truth, **kwargs) -> List[float]:
    """
    Macro recall over GT categories at IoU=0.5.

    This directly penalizes the current failure mode where the policy learns to
    report easy classes but drops hard ones such as backpack_box.
    """
    rewards: List[float] = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        parsed = _parse_prediction(text)
        gts = gt.get("detections", []) if isinstance(gt, dict) else []

        if parsed is None:
            rewards.append(0.0)
            continue

        preds_valid = [pred for pred in parsed if _is_valid_detection(pred)]

        if len(gts) == 0:
            rewards.append(0.0)
            continue
        if len(preds_valid) == 0:
            rewards.append(0.0)
            continue

        matches, _, _ = _match_predictions(
            preds_valid,
            gts,
            iou_threshold=0.5,
            label_compatible_fn=_eval_aligned_label_compatible,
        )
        gt_total_by_cat: Dict[Tuple[Optional[str], Optional[str]], int] = {}
        gt_matched_by_cat: Dict[Tuple[Optional[str], Optional[str]], int] = {}

        for gt_det in gts:
            key = _category_key(gt_det)
            gt_total_by_cat[key] = gt_total_by_cat.get(key, 0) + 1
            gt_matched_by_cat.setdefault(key, 0)

        for _, gt_idx, _ in matches:
            key = _category_key(gts[gt_idx])
            gt_matched_by_cat[key] = gt_matched_by_cat.get(key, 0) + 1

        per_category_recalls = [
            gt_matched_by_cat[key] / total
            for key, total in gt_total_by_cat.items()
            if total > 0
        ]
        rewards.append(sum(per_category_recalls) / len(per_category_recalls))
    return rewards


def build_v2_reward_funcs():
    """Return (funcs, weights) for the v2 reward scheme."""
    funcs = [
        v2_format_reward,
        v2_detection_f1_reward,
        v2_state_reward,
        v2_coverage_reward,
    ]
    weights = [1.0, 3.0, 2.0, 2.0]
    return funcs, weights


def build_simple_reward_funcs():
    """Return a small reward bundle aligned with AP50-style detection quality."""
    funcs = [
        simple_format_reward,
        simple_detection_reward,
        simple_state_reward,
        simple_category_recall_reward,
    ]
    weights = [0.2, 4.0, 1.0, 1.5]
    return funcs, weights


if __name__ == "__main__":
    # Tiny self-test.
    gt = [{
        "detections": [
            {
                "device_type": "traffic_signal",
                "sub_type": "vehicle_signal",
                "state": "normal",
                "bbox_1000": [100, 100, 200, 200],
            }
        ]
    }]
    diff = [{"num_targets": 1, "num_abnormal": 0, "has_tiny_object": False, "empty_scene": False, "has_ignore_region": False}]
    completion = ['{"detections":[{"device_type":"traffic_signal","sub_type":"vehicle_signal","state":"normal","bbox_1000":[100,100,200,200]}]}']
    print("combined:", combined_reward_func(completions=completion, ground_truth=gt, difficulty=diff))
