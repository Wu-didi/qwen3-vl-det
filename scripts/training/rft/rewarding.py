"""RFT 奖励函数与解析辅助工具。"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

NO_DETECTION_PATTERNS = [
    "未检测到相关设备",
    "未检测到设备",
    "未检测到目标",
    "没有检测到",
    "no relevant equipment",
    "no equipment detected",
]

BOX_PATTERN = (
    r"<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>"
)

ANOMALY_KEYWORDS = ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"]


@dataclass
class ParsedDetection:
    """从模型文本中解析出的结构化检测结果。"""

    category: str
    status: str
    is_anomaly: bool
    bbox: List[int]


@dataclass
class RiskRewardConfig:
    """风险感知奖励的超参数配置。"""

    match_iou_threshold: float = 0.5
    hallucination_unit_penalty: float = 0.35
    no_detection_missing_penalty: float = 0.2
    omission_penalty: float = 1.0


RISK_REWARD_CFG = RiskRewardConfig()


def set_risk_reward_config(cfg: RiskRewardConfig) -> None:
    """更新全局风险奖励配置（供各奖励函数共享）。"""
    global RISK_REWARD_CFG
    RISK_REWARD_CFG = cfg


def get_risk_reward_config() -> RiskRewardConfig:
    """返回当前风险奖励配置。"""
    return RISK_REWARD_CFG


def _is_no_detection_response(text: str) -> bool:
    """判断模型是否明确输出“未检测到设备”语义。"""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in NO_DETECTION_PATTERNS)


def _has_structured_detection_format(text: str) -> bool:
    """检查输出是否满足“序号 + 状态 + 坐标框”的结构化格式。"""
    has_box = bool(re.search(BOX_PATTERN, text))
    has_numbered = bool(re.search(r"\d+\.\s+\S+", text))
    has_status = bool(re.search(r"状态[：:]\s*\S+", text))
    return has_box and has_numbered and has_status


def _is_format_valid(completion: str, gt_response: Optional[str] = None) -> bool:
    """奖励函数统一使用的严格格式门控。"""
    if _has_structured_detection_format(completion):
        return True

    if gt_response is None:
        return False

    gt_boxes = _extract_boxes(gt_response)
    if not gt_boxes and _is_no_detection_response(completion):
        return True

    return False


def format_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """格式奖励（门控项）：合法=1，不合法=0。"""
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = assistant[idx] if assistant and idx < len(assistant) else None
        rewards.append(1.0 if _is_format_valid(completion, gt_response) else 0.0)

    return rewards


def bbox_iou_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """基于 IoU 的定位奖励。"""
    rewards = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_boxes = _extract_boxes(completion)
        gt_boxes = _extract_boxes(gt_response)

        if not gt_boxes:
            if not pred_boxes and _is_no_detection_response(completion):
                reward = 1.0
            else:
                reward = 0.0
        elif not pred_boxes:
            reward = 0.0
        else:
            ious = []
            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in pred_boxes:
                    iou = _compute_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)
            reward = sum(ious) / len(ious) if ious else 0.0

        rewards.append(reward)

    return rewards


def category_match_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """类别匹配奖励。"""
    rewards = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_cats = _extract_categories(completion)
        gt_cats = _extract_categories(gt_response)

        if not gt_cats:
            reward = 1.0 if (not pred_cats and _is_no_detection_response(completion)) else 0.0
        elif not pred_cats:
            reward = 0.0
        else:
            matches = 0
            for gt_cat in gt_cats:
                for pred_cat in pred_cats:
                    if gt_cat in pred_cat or pred_cat in gt_cat:
                        matches += 1
                        break
            reward = matches / len(gt_cats)

        rewards.append(reward)

    return rewards


def status_accuracy_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """状态准确率奖励（正常/异常）。"""
    rewards = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_statuses = _extract_statuses(completion)
        gt_statuses = _extract_statuses(gt_response)

        if not gt_statuses:
            reward = 1.0 if (not pred_statuses and _is_no_detection_response(completion)) else 0.0
        elif not pred_statuses:
            reward = 0.0
        else:
            pred_has_anomaly = any(any(kw in s for kw in ANOMALY_KEYWORDS) for s in pred_statuses)
            gt_has_anomaly = any(any(kw in s for kw in ANOMALY_KEYWORDS) for s in gt_statuses)
            reward = 1.0 if pred_has_anomaly == gt_has_anomaly else 0.0

        rewards.append(reward)

    return rewards


def set_f1_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """集合级检测 F1 奖励（基于一对一 TP/FP/FN 匹配）。"""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        tp = len(matches)
        fp = max(pred_count - tp, 0)
        fn = max(gt_count - tp, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rewards.append(_safe_f1(precision, recall))

    return rewards


def localization_quality_reward(
    completions: List[str], assistant: List[str], **kwargs
) -> List[float]:
    """定位质量奖励：对匹配对计算 IoU 并归一化。"""
    rewards: List[float] = []
    iou_thr = RISK_REWARD_CFG.match_iou_threshold
    denom = max(1e-6, 1.0 - iou_thr)

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)

        if not gt_dets:
            if not pred_dets and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=iou_thr,
            require_category=True,
        )

        if not matches:
            rewards.append(0.0)
            continue

        mean_iou = sum(m["iou"] for m in matches) / len(matches)
        norm_iou = max(0.0, min(1.0, (mean_iou - iou_thr) / denom))
        rewards.append(norm_iou)

    return rewards


def count_alignment_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """计数一致性奖励：惩罚明显过检和漏检。"""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_count = len(_extract_detections(completion))
        gt_count = len(_extract_detections(gt_response))

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-penalty)
            continue

        if _is_no_detection_response(completion) and pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        ratio_err = abs(pred_count - gt_count) / max(gt_count, 1)
        rewards.append(max(-1.0, min(1.0, 1.0 - ratio_err)))

    return rewards


def risk_control_reward(completions: List[str], assistant: List[str], **kwargs) -> List[float]:
    """风险控制奖励：惩罚空场景误报与正样本漏检。"""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)
        no_detection = _is_no_detection_response(completion)

        if gt_count == 0:
            if pred_count == 0 and no_detection:
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                halluc_penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-halluc_penalty)
            continue

        if pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=False,
        )
        covered_gt = {m["gt_idx"] for m in matches}
        covered_pred = {m["pred_idx"] for m in matches}

        omission_rate = 1.0 - (len(covered_gt) / max(gt_count, 1))
        hallucinated = max(pred_count - len(covered_pred), 0)
        reward = (
            1.0
            - RISK_REWARD_CFG.omission_penalty * omission_rate
            - min(1.0, hallucinated * RISK_REWARD_CFG.hallucination_unit_penalty)
        )
        if no_detection:
            reward -= RISK_REWARD_CFG.omission_penalty

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def anomaly_instance_f1_reward(
    completions: List[str], assistant: List[str], **kwargs
) -> List[float]:
    """实例级异常识别奖励（略偏向召回）。"""
    rewards: List[float] = []

    for completion, gt_response in zip(completions, assistant):
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _extract_detections(gt_response)
        no_detection = _is_no_detection_response(completion)

        if not gt_dets:
            if not pred_dets and no_detection:
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                false_alarm = sum(1 for det in pred_dets if det.is_anomaly)
                rewards.append(-min(1.0, 0.5 * false_alarm))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        if not matches:
            gt_has_anomaly = any(det.is_anomaly for det in gt_dets)
            rewards.append(-1.0 if gt_has_anomaly else 0.0)
            continue

        matched_pred_indices = {m["pred_idx"] for m in matches}
        gt_anomaly_total = sum(1 for det in gt_dets if det.is_anomaly)
        pred_anomaly_total = 0
        tp_anomaly = 0

        for m in matches:
            pred_anomaly = pred_dets[m["pred_idx"]].is_anomaly
            gt_anomaly = gt_dets[m["gt_idx"]].is_anomaly
            if pred_anomaly:
                pred_anomaly_total += 1
            if pred_anomaly and gt_anomaly:
                tp_anomaly += 1

        unmatched_pred_anomaly = sum(
            1
            for idx, det in enumerate(pred_dets)
            if idx not in matched_pred_indices and det.is_anomaly
        )
        fp_anomaly = max(pred_anomaly_total - tp_anomaly, 0) + unmatched_pred_anomaly
        fn_anomaly = max(gt_anomaly_total - tp_anomaly, 0)

        if gt_anomaly_total == 0:
            rewards.append(1.0 if fp_anomaly == 0 else max(-1.0, 1.0 - 0.5 * fp_anomaly))
            continue

        if no_detection and len(pred_dets) == 0:
            rewards.append(-1.0)
            continue

        precision = tp_anomaly / (tp_anomaly + fp_anomaly) if (tp_anomaly + fp_anomaly) > 0 else 0.0
        recall = tp_anomaly / (tp_anomaly + fn_anomaly) if (tp_anomaly + fn_anomaly) > 0 else 0.0
        reward = 0.4 * precision + 0.6 * recall
        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def _extract_boxes(text: str) -> List[List[int]]:
    """从文本中提取边界框，支持整数/浮点数/负数坐标。"""
    boxes = []
    for match in re.finditer(BOX_PATTERN, text):
        try:
            box = [int(float(match.group(i))) for i in range(1, 5)]
            boxes.append(box)
        except (ValueError, TypeError):
            continue
    return boxes


def _extract_categories(text: str) -> List[str]:
    """从编号条目中提取类别文本。"""
    categories = []
    pattern = r"\d+\.\s*([^\n]+)"
    for match in re.finditer(pattern, text):
        cat = match.group(1).strip()
        if "状态" in cat:
            cat = cat.split("状态")[0].strip()
        if cat:
            categories.append(cat)
    return categories


def _extract_statuses(text: str) -> List[str]:
    """提取“状态: xxx”字段。"""
    statuses = []
    pattern = r"状态[：:]\s*([^\n]+)"
    for match in re.finditer(pattern, text):
        statuses.append(match.group(1).strip())
    return statuses


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的 IoU。"""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _normalize_category(category: str) -> str:
    """类别文本标准化：去空白并转小写。"""
    return re.sub(r"\s+", "", category.strip().lower())


def _is_anomaly_status(status: str) -> bool:
    """通过关键词判断是否为异常状态。"""
    return any(kw in status for kw in ANOMALY_KEYWORDS)


def _category_match(pred_category: str, gt_category: str) -> bool:
    """类别模糊匹配（完全相等或包含关系）。"""
    pred = _normalize_category(pred_category)
    gt = _normalize_category(gt_category)
    if not pred or not gt:
        return False
    if pred == "unknown" or gt == "unknown":
        return False
    return pred == gt or pred in gt or gt in pred


def _extract_detections(text: str) -> List[ParsedDetection]:
    """从输出文本解析结构化检测项（类别/状态/框）。"""
    detections: List[ParsedDetection] = []
    items = re.split(r"(?=\d+\.\s+)", text)

    for item in items:
        if not item.strip():
            continue

        cat_match = re.match(r"(\d+)\.\s*([^\n]+)", item)
        if not cat_match:
            continue

        category = cat_match.group(2).strip()
        status_match = re.search(r"状态[：:]\s*([^\n]+)", item)
        status = status_match.group(1).strip() if status_match else "正常"

        box_match = re.search(BOX_PATTERN, item)
        if not box_match:
            continue

        try:
            bbox = [int(float(box_match.group(i))) for i in range(1, 5)]
        except (ValueError, TypeError):
            continue

        detections.append(
            ParsedDetection(
                category=category,
                status=status,
                is_anomaly=_is_anomaly_status(status),
                bbox=bbox,
            )
        )

    if not detections:
        for box in _extract_boxes(text):
            detections.append(
                ParsedDetection(
                    category="unknown",
                    status="unknown",
                    is_anomaly=False,
                    bbox=box,
                )
            )

    return detections


def _match_detections(
    pred_dets: List[ParsedDetection],
    gt_dets: List[ParsedDetection],
    iou_threshold: float,
    require_category: bool = True,
) -> List[Dict[str, Any]]:
    """基于 IoU 的贪心一对一匹配。"""
    candidates: List[Tuple[float, int, int, bool]] = []
    for pred_idx, pred_det in enumerate(pred_dets):
        for gt_idx, gt_det in enumerate(gt_dets):
            iou = _compute_iou(pred_det.bbox, gt_det.bbox)
            if iou < iou_threshold:
                continue
            category_ok = _category_match(pred_det.category, gt_det.category)
            if require_category and not category_ok:
                continue
            score = iou + (0.01 if category_ok else 0.0)
            candidates.append((score, pred_idx, gt_idx, category_ok))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    matches: List[Dict[str, Any]] = []

    for _, pred_idx, gt_idx, category_ok in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        iou = _compute_iou(pred_dets[pred_idx].bbox, gt_dets[gt_idx].bbox)
        matches.append(
            {
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou,
                "category_ok": category_ok,
            }
        )

    return matches


def _safe_f1(precision: float, recall: float) -> float:
    """安全版 F1 计算，分母为 0 时返回 0。"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
