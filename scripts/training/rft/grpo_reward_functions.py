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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def _normalize_sub_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "null":
            return None
        return value
    return str(value)


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
        device_type = det.get("device_type")
        sub_type = _normalize_sub_type(det.get("sub_type"))
        state = det.get("state")
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

    if device_type not in ALLOWED_DEVICE_TYPES:
        return False
    if sub_type not in ALLOWED_SUB_TYPES:
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


def _match_predictions(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy one-to-one matching under label compatibility and IoU threshold.
    Returns:
        matches: list of (pred_idx, gt_idx, iou)
        unmatched_pred_indices
        unmatched_gt_indices
    """
    candidates: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        if pred.get("bbox_1000") is None:
            continue
        for gi, gt in enumerate(gts):
            if not _label_compatible(pred, gt):
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
